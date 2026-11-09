import struct

def update(state, item):
    state = bytearray(state)
    if item < 128:
        # Exact counter for 0-127
        # Offset is item
        current = state[item]
        state[item] = (current + 1) % 256
    else:
        # Sketch for 128-399
        # 4 rows, 224 columns, uint8
        # Base offset 128
        # Row stride 224
        ps = (1035493289, 775374203, 1495862053, 2015145641)
        for r in range(4):
            p = ps[r]
            idx = (item * p + 31 * r) % 224
            offset = 128 + r * 224 + idx
            current = state[offset]
            state[offset] = (current + 1) % 256
    return bytes(state)

def query(state, item):
    if item < 128:
        return state[item]
    else:
        ps = (1035493289, 775374203, 1495862053, 2015145641)
        min_val = 255
        for r in range(4):
            p = ps[r]
            idx = (item * p + 31 * r) % 224
            offset = 128 + r * 224 + idx
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
