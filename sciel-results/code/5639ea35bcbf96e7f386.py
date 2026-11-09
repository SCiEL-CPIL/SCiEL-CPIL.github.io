import struct

# The domain is 0..399.
# We allocate 400 uint16 counters.
# 400 * 2 bytes = 800 bytes.
# The state is 1024 bytes, so the rest remains 0.

def update(state, item):
    buf = bytearray(state)
    # item is in 0..399
    # Offset for item 'item' is item * 2
    offset = item * 2
    # Read current value (little-endian uint16)
    val = struct.unpack_from('<H', buf, offset)[0]
    # Increment
    val += 1
    # Write back
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    # item is in 0..399
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
