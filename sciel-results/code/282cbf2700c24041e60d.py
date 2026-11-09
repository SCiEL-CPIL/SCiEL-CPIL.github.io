import struct

# Hybrid: 64 exact uint8 counters for items 0..63
# + 4 rows x 120 uint16 CMS counters for items 64..399
# State: bytes 0..63 = exact, bytes 64..1023 = 4*120*2 = 960 bytes of sketch

EXACT_N = 64
SKETCH_ROWS = 4
SKETCH_W = 120  # (1024 - 64) / 2 / 4 = 120
PRIME = 2147483647
_P = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute sketch indices for items 64..399
_SK_IDX = [[0] * SKETCH_ROWS for _ in range(400)]
for item in range(EXACT_N, 400):
    for r in range(SKETCH_ROWS):
        _SK_IDX[item][r] = ((item * _P[r] + 31 * r) % PRIME) % SKETCH_W

def update(state, item):
    buf = bytearray(state)
    if item < EXACT_N:
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        idxs = _SK_IDX[item]
        for r in range(SKETCH_ROWS):
            idx = idxs[r]
            offset = EXACT_N + r * SKETCH_W * 2 + idx * 2
            cur = int.from_bytes(buf[offset:offset+2], 'little')
            if cur < 65535:
                buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    return bytes(buf)

def query(state, item):
    if item < EXACT_N:
        return state[item]
    else:
        idxs = _SK_IDX[item]
        min_val = 65535
        for r in range(SKETCH_ROWS):
            idx = idxs[r]
            offset = EXACT_N + r * SKETCH_W * 2 + idx * 2
            val = int.from_bytes(state[offset:offset+2], 'little')
            if val < min_val:
                min_val = val
        return min_val
