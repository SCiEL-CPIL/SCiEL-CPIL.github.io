import struct

# Hybrid: exact counters for items 0-19, sketch for items 20-399
W = 501
D = 2
PRIME = 2147483647
_PS = (1035493289, 775374203)

# Exact counter layout:
# Item 0: bytes 0-1 (uint16 LE)
# Items 1-19: bytes 2-20 (1 byte each)
# Sketch: bytes 21-1023 (1003 bytes, 2 rows x 501)

SKETCH_OFFSET = 21

_INDICES = [[0] * D for _ in range(400)]
for item in range(400):
    for r in range(D):
        p = _PS[r]
        _INDICES[item][r] = ((item * p + 31 * r) % PRIME) % W

def update(state, item):
    buf = bytearray(state)
    if item == 0:
        # uint16 LE at offset 0
        val = struct.unpack_from('<H', buf, 0)[0]
        if val < 65535:
            val += 1
            struct.pack_into('<H', buf, 0, val)
    elif item <= 19:
        # uint8 at offset 1 + item
        offset = 1 + item
        cur = buf[offset]
        if cur < 255:
            buf[offset] = cur + 1
    # Update sketch for all items (including 0-19, but query will use exact)
    idxs = _INDICES[item]
    for r in range(D):
        offset = SKETCH_OFFSET + r * W + idxs[r]
        cur = buf[offset]
        if cur < 255:
            buf[offset] = cur + 1
    return bytes(buf)

def query(state, item):
    if item == 0:
        return struct.unpack_from('<H', state, 0)[0]
    elif item <= 19:
        return state[1 + item]
    else:
        idxs = _INDICES[item]
        vals = []
        for r in range(D):
            offset = SKETCH_OFFSET + r * W + idxs[r]
            vals.append(state[offset])
        return min(vals)
