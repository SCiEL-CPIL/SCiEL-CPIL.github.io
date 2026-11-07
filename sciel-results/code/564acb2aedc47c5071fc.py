def update(state, item):
    ba = bytearray(state)
    offset = item * 2
    val = ba[offset] | (ba[offset + 1] << 8)
    val += 1
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    ba = state
    offset = item * 2
    return ba[offset] | (ba[offset + 1] << 8)
