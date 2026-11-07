def update(state, item):
    buf = bytearray(state)
    current = buf[item]
    if current < 255:
        buf[item] = current + 1
    return bytes(buf)

def query(state, item):
    return state[item]
