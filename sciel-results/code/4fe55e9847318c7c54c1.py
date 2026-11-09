import struct

# Hybrid Frequency Sketch:
# 1. Exact counters for items 0..255 (2 bytes each, 512 bytes total)
# 2. Count-Min Sketch (1 row, 256 slots, uint16) for items 256..399 (512 bytes total)

W_SKETCH = 256
PRIME = 2147483647
_P = 1035493289

# Precompute sketch indices for items 256..399
# _SKETCH_INDICES[item] = index in 0..255
_SKETCH_INDICES = [0] * 400
for item in range(256, 400):
    _SKETCH_INDICES[item] = ((item * _P) % PRIME) % W_SKETCH

def update(state, item):
    buf = bytearray(state)
    if item < 256:
        offset = item * 2
        cur = int.from_bytes(buf[offset:offset+2], 'little')
        if cur < 65535:
            buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    else:
        idx = _SKETCH_INDICES[item]
        offset = 512 + idx * 2
        cur = int.from_bytes(buf[offset:offset+2], 'little')
        if cur < 65535:
            buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    return bytes(buf)

def query(state, item):
    if item < 256:
        offset = item * 2
        return int.from_bytes(state[offset:offset+2], 'little')
    else:
        idx = _SKETCH_INDICES[item]
        offset = 512 + idx * 2
        return int.from_bytes(state[offset:offset+2], 'little')
