import struct

# EVOLVE-BLOCK-START
# Structural Change: Increase depth to 16, decrease width to 32.
# Total counters: 16 * 32 = 512 uint16 counters = 1024 bytes.
# This reduces the probability of a heavy hitter colliding with a rare item
# in ALL rows, thus lowering the systematic over-estimation bias.

DEPTH = 16
WIDTH = 32
COUNTER_SIZE = 2

# Precompute index table for items 0..399
# Use a simple multiplicative hash with good mixing.
# (item * prime + salt) & (WIDTH - 1)
# Since WIDTH is 32 (power of 2), we use & 31.

PRIMES = [
    2654435761,
    40503,
    69069,
    2862933555777941757 % 2**32,
    3037000493,
    2246822519,
    3266489917,
    668265263,
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2285881789,
    2473401169,
    2600822929,
    2807566739
]

ROW_SALTS = [
    0x9e3779b9, 0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f,
    0x165667b1, 0x9e3779b9, 0x85ebca6b, 0xc2b2ae35,
    0x27d4eb2f, 0x165667b1, 0x9e3779b9, 0x85ebca6b,
    0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0x9e3779b9
]

_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(DEPTH):
        h = (item * PRIMES[r] + ROW_SALTS[r]) & 0xFFFFFFFF
        h ^= (h >> 16)
        h &= 0xFFFF
        row_indices.append(h % WIDTH)
    _IDX_TABLE.append(row_indices)

def update(state, item):
    b = bytearray(state)
    item_indices = _IDX_TABLE[item]
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val >= 65536:
            val = 65535
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    item_indices = _IDX_TABLE[item]
    min_val = 65535
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
