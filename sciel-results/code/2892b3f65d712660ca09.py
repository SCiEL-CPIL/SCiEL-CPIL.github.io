import numpy as np

# ... inside construct ...
offsets_np = np.array(offsets, dtype=np.int32)

# inside loop:
forbidden_np = np.zeros(MAX_WORDS, dtype=np.uint8)
# ...
for i in order:
    if forbidden_np[i]:
        continue
    code.append(i)
    forbidden_np[i] = 1
    forbidden_np[i ^ offsets_np] = 1
