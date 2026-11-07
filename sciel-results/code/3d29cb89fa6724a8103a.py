"""Direct-mapped 16-bit frequency table for all 400 items (exact counting)."""
# EVOLVE-BLOCK-START

import struct


def update(state, item):
    """Increment the 16-bit counter for the given item."""
    state = bytearray(state)
    pos = item * 2
    # Read current 16-bit little-endian counter
    val = state[pos] | (state[pos + 1] << 8)
    # Increment (max possible is 1024, fits in 16 bits)
    val += 1
    # Write back
    state[pos] = val & 0xFF
    state[pos + 1] = (val >> 8) & 0xFF
    return bytes(state)


def query(state, item):
    """Return the exact 16-bit count for the given item."""
    pos = item * 2
    return state[pos] | (state[pos + 1] << 8)

# EVOLVE-BLOCK-END
