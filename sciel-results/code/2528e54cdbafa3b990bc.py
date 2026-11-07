import struct

# Count-Min Sketch: 4 rows x 128 columns of uint16 counters = 1024 bytes.
# Precomputed index table for items 0..399 to speed up update/query.
# Uses well-distributed hash primes with a clean modulo-128 mapping.

DEPTH = 4
WIDTH = 128
COUNTER_SIZE = 2

# Primes chosen for good distribution
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
]

# Precompute the 400 x 4 index table
_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(DEPTH):
        # Use the seed's hash structure with width 128
        idx = ((item * PRIMES[r] + 31 * r) % 2147483647) % WIDTH
        row_indices.append(idx)
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
