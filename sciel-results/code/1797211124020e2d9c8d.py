def update(state, item):
    b = bytearray(state)
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    val = int.from_bytes(b[offset:offset+2], 'little')
    val += 1
    b[offset:offset+2] = val.to_bytes(2, 'little')
    return bytes(b)

def query(state, item):
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    return val
