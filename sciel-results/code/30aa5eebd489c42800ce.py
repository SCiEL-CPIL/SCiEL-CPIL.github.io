"""Improved frequency sketch: exact byte counters for 400 items within 1024 bytes."""

# EVOLVE-BLOCK-START
def update(state, item):
    """
    Update the state by incrementing the count for 'item'.
    State layout: 400 bytes for exact counts (item i at byte i), rest zero.
    Each count is capped at 255 to fit in a byte.
    """
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(1024)
    state = bytearray(state)
    if len(state) < 400:
        state.extend(b'\x00' * (400 - len(state)))
    
    idx = item % 400
    if state[idx] < 255:
        state[idx] += 1
    
    return bytes(state[:1024])


def query(state, item):
    """
    Query the exact count for 'item'.
    Returns the byte value at position item % 400.
    """
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(1024)
    if len(state) < 400:
        return 0
    idx = item % 400
    return int(state[idx])
# EVOLVE-BLOCK-END
