def update(state, item):
    b = bytearray(state)
    if 0 <= item < 400:
        offset = item * 2
        # Read current uint16
        val = b[offset] | (b[offset + 1] << 8)
        val += 1
        # Write back uint16
        b[offset] = val & 0xFF
        b[offset + 1] = (val >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    if 0 <= item < 400:
        offset = item * 2
        return state[offset] | (state[offset + 1] << 8)
    return 0
