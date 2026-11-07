"""Exact 2-byte counter per item: 400 items fit in 800 bytes, leaving 224 bytes unused."""
# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    if item < 400:
        offset = item * 2
        # Read current 16-bit little-endian counter
        current = state[offset] | (state[offset + 1] << 8)
        # Increment
        new_val = current + 1
        # Write back 16-bit little-endian
        state[offset] = new_val & 0xFF
        state[offset + 1] = (new_val >> 8) & 0xFF
    return bytes(state)

def query(state, item):
    if item < 400:
        offset = item * 2
        val = state[offset] | (state[offset + 1] << 8)
        return float(val)
    return 0.0
# EVOLVE-BLOCK-END
