"""Hybrid exact counter: 400 uint16 counters in 1024-byte state."""

def update(state, item):
    # state is 1024 bytes; first 800 bytes hold 400 uint16 little-endian counters
    # item is in 0..399
    offset = item * 2
    # Read current value
    val = state[offset] | (state[offset + 1] << 8)
    # Increment, cap at 65535 (won't be reached since max is 1024)
    val += 1
    # Write back
    new_state = bytearray(state)
    new_state[offset] = val & 0xFF
    new_state[offset + 1] = (val >> 8) & 0xFF
    return bytes(new_state)

def query(state, item):
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
