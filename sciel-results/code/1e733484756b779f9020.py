import struct

# EVOLVE-BLOCK-START
# Structural Change: Dense 16-bit Exact Counters
# We have 400 items (0..399).
# We allocate 2 bytes (16-bit unsigned integer) for each item.
# Total state size: 400 * 2 = 800 bytes.
# This is within the 1024 byte limit.
# This allows us to track the exact count of every item in all streams.
# Since the maximum possible count is 3072 (3 streams * 1024 items),
# 16-bit counters (max 65535) are sufficient and will not overflow.
# This eliminates all estimation error, resulting in a perfect score.

ITEM_COUNT = 400
BYTES_PER_ITEM = 2

def update(state, item):
    s = bytearray(state)
    if 0 <= item < ITEM_COUNT:
        pos = item * BYTES_PER_ITEM
        # Read 16-bit little-endian
        val = struct.unpack_from('<H', s, pos)[0]
        val += 1
        # Write back
        struct.pack_into('<H', s, pos, val)
    return bytes(s)

def query(state, item):
    if 0 <= item < ITEM_COUNT:
        pos = item * BYTES_PER_ITEM
        val = struct.unpack_from('<H', state, pos)[0]
        return val
    else:
        return 0
# EVOLVE-BLOCK-END
