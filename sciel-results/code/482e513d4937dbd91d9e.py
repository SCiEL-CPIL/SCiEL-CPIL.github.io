"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def update(state, item):
    b = bytearray(state)
    idx = item * 2
    # Read uint16 little-endian
    val = b[idx] | (b[idx + 1] << 8)
    val += 1
    # Write back
    b[idx] = val & 0xFF
    b[idx + 1] = (val >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    idx = item * 2
    val = state[idx] | (state[idx + 1] << 8)
    return val
# EVOLVE-BLOCK-END
