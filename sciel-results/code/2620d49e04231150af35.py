def update(state, item):
    b = bytearray(state)
    # Golden ratio hash
    h = (item * 2654435761) & 0xFFFFFFFF
    # Use upper bits to index 512 buckets (0..511)
    idx = (h >> 16) & 0x1FF
    
    offset = idx * 2
    val = int.from_bytes(b[offset:offset+2], 'little')
    val += 1
    b[offset:offset+2] = val.to_bytes(2, 'little')
    return bytes(b)

def query(state, item):
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 16) & 0x1FF
    offset = idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    return val
