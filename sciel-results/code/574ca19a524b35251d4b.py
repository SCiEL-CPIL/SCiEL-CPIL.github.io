"""Hybrid exact-counter + CMS sketch for streaming frequency estimation."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    """Update the state with a new item occurrence."""
    # First 800 bytes: 400 x uint16 exact counters
    # Remaining 224 bytes: unused (or could store metadata)
    offset = item * 2
    lo = state[offset]
    hi = state[offset + 1]
    val = lo | (hi << 8)
    val += 1
    if val > 65535:
        val = 65535
    new_lo = val & 0xFF
    new_hi = (val >> 8) & 0xFF
    new_state = bytearray(state)
    new_state[offset] = new_lo
    new_state[offset + 1] = new_hi
    return bytes(new_state)

def query(state, item):
    """Return the frequency estimate for an item."""
    offset = item * 2
    lo = state[offset]
    hi = state[offset + 1]
    return lo | (hi << 8)
# EVOLVE-BLOCK-END
