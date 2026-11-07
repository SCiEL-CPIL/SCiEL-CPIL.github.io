import struct

_NUM_ITEMS = 400
# State layout:
# Bytes 0-799: 400 x uint16 (little-endian) exact counters
# Bytes 800-1023: Unused (or could be used for a small sketch, but not needed)

def update(state, item):
    """Increment the dedicated uint16 counter for this item."""
    buf = bytearray(state)
    offset = item * 2
    # Read uint16
    val = struct.unpack_from('<H', buf, offset)[0]
    # Increment
    val += 1
    # Write back
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    """Return exact count from dedicated uint16 counter."""
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]
