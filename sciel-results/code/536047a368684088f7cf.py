def update(state, item):
    new_state = bytearray(state)
    offset = item * 2
    # Extract current value
    val = new_state[offset] | (new_state[offset+1] << 8)
    val += 1
    # Write back
    new_state[offset] = val & 0xFF
    new_state[offset+1] = (val >> 8) & 0xFF
    return bytes(new_state)

def query(state, item):
    offset = item * 2
    return state[offset] | (state[offset+1] << 8)
