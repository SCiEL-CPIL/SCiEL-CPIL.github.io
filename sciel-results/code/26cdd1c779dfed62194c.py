# ... (imports) ...

# Precompute GF(2^8) multiplication table for poly 0x11D
GF_TABLE = np.zeros((256, 256), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        # ... compute ...
