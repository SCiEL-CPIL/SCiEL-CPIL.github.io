import struct

# Exact frequency counting: 400 items, each with a 2-byte counter (uint16)
# State layout: 800 bytes (2 bytes per item) + 224 bytes of padding (zeros)
# Total: 1024 bytes

def update(state, item):
    """
    Updates the exact frequency counter for 'item' in the state.
    State layout: 400 uint16 counters (one per item 0..399) + padding.
    """
    ba = bytearray(state)
    offset = item * 2
    # Read current value
    val = struct.unpack_from('<H', ba, offset)[0]
    # Increment and write back
    val += 1
    struct.pack_into('<H', ba, offset, val)
    return bytes(ba)

def query(state, item):
    """
    Queries the exact frequency of 'item'.
    """
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]
