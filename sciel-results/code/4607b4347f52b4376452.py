def update(state, item):
    """
    Increment the count for 'item' in the state.
    state: bytes of length 1024
    item: int in 0..399
    Returns new state as bytes of length 1024.
    """
    ba = bytearray(state)
    offset = item * 2
    # Read current count (little-endian uint16)
    current = ba[offset] | (ba[offset + 1] << 8)
    if current < 65535:
        current += 1
    # Write back
    ba[offset] = current & 0xFF
    ba[offset + 1] = (current >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    """
    Return the exact frequency estimate for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns nonnegative int.
    """
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
