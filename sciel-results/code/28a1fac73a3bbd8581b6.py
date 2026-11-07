def update(state, item):
    b = bytearray(state)
    # 4 rows x 128 uint16 counters = 1024 bytes
    # Row r starts at byte offset r*256, each counter is 2 bytes
    # Hash: use different multipliers per row
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    for r in range(4):
        h = (item * multipliers[r]) & 0xFFFFFFFF
        idx = h % 128  # 128 counters per row
        offset = r * 256 + idx * 2
        val = int.from_bytes(b[offset:offset+2], 'little')
        val = (val + 1) & 0xFFFF
        b[offset:offset+2] = val.to_bytes(2, 'little')
    return bytes(b)

def query(state, item):
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    min_val = 0xFFFF
    for r in range(4):
        h = (item * multipliers[r]) & 0xFFFFFFFF
        idx = h % 128
        offset = r * 256 + idx * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
    return min_val
