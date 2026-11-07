import struct

# EVOLVE-BLOCK-START
# Structural Change: Replace probabilistic Count-Min Sketch with an exact frequency table.
# Since items are in 0..399 and stream length is 1024, max count is 1024.
# We use 400 uint16 counters (800 bytes) within the 1024-byte state.
# This provides exact counts, eliminating all collision error.

def update(state, item):
    b = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', b, offset)[0]
    val += 1
    # val will not exceed 1024, so no overflow handling needed for uint16
    struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
# EVOLVE-BLOCK-END
