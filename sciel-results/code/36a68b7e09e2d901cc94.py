import struct

# Count-Min Sketch with 1 row of 512 uint16 counters
W = 512
D = 1
PRIME = 2147483647
# Using the first prime from the seed for the single row
_P = 1035493289

# Precompute indices
_INDICES = [0] * 400
for item in range(400):
    _INDICES[item] = ((item * _P + 31 * 0) % PRIME) % W

def update(state, item):
    buf = bytearray(state)
    idx = _INDICES[item]
    offset = idx * 2
    cur = int.from_bytes(buf[offset:offset+2], 'little')
    if cur < 65535:
        buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    return bytes(buf)

def query(state, item):
    idx = _INDICES[item]
    offset = idx * 2
    return int.from_bytes(state[offset:offset+2], 'little')
