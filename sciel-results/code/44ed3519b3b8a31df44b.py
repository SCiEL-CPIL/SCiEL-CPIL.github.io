"""Exact frequency sketch using 1024-byte state for 400 items."""

def update(state, item):
    """
    Increment the count for 'item' in the state.
    state: bytes of length 1024
    item: int in 0..399
    Returns new state as bytes of length 1024.
    """
    ba = bytearray(state)
    idx = item * 2
    current = ba[idx] | (ba[idx + 1] << 8)
    if current < 65535:
        current += 1
        ba[idx] = current & 0xFF
        ba[idx + 1] = (current >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    """
    Return the exact frequency estimate for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns nonnegative int.
    """
    idx = item * 2
    return state[idx] | (state[idx + 1] << 8)
