import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.constants import IMAGE_TOKEN_INDEX

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model info
model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)

# Load model (do NOT specify offload_folder if you're using GPU directly)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b",
    model_name="llava-v1.5-7b",
    model_base=None,
    load_8bit=False,
    load_4bit=False
)

# --- Load local image ---
image_path = "frame_750.jpg"  # ← replace with your image filename
image = Image.open(image_path).convert("RGB")
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()


# --- Prompt ---
prompt = "What are the things I should be cautious about when I visit here?"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
print(input_ids, type(model))

# --- Generate ---
with torch.no_grad():
    output_ids = model.generate(
        inputs=input_ids,
        do_sample=True,
        images=image_tensor,
        max_new_tokens=512
    )

    # --- Decode ---
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output)