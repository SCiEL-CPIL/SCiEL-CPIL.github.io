"""Exact 1-byte counter per item: 400 items fit in 1024 bytes."""
# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    # Use item index directly as offset. Items are 0..399.
    # Each counter is 1 byte (0-255).
    # If item < 400, we have a dedicated byte.
    # If item >= 400 (should not happen per contract, but safe), fallback to CM-like or ignore.
    if item < 400:
        val = state[item]
        if val < 255:
            state[item] = val + 1
        # else: saturated at 255, do nothing (count is >= 255)
    return bytes(state)

def query(state, item):
    if item < 400:
        return float(state[item])
    return 0.0
# EVOLVE-BLOCK-END
