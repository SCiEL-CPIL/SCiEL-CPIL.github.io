import struct

def update(state, item):
    """
    Updates the frequency count for the given item in the state.
    State layout:
    - Bytes 0-799: 400 uint16 counters (little-endian), one for each item 0-399.
    - Bytes 800-1023: Padding (unused).
    """
    state = bytearray(state)
    offset = item * 2
    # Read current count
    count = struct.unpack_from('<H', state, offset)[0]
    # Increment
    count += 1
    # Write back
    struct.pack_into('<H', state, offset, count)
    return bytes(state)

def query(state, item):
    """
    Returns the exact frequency count for the given item.
    """
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    return count
