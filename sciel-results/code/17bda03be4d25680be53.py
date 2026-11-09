# Variable-depth CMS: items 0-199 use 4 rows, items 200-399 use 2 rows
# All rows have width 256, using uint8 counters
# Total: 4 * 256 = 1024 bytes

W = 256
D_HIGH = 4   # rows for items 0-199
D_LOW = 2    # rows for items 200-399
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
PRIME = 2147483647
SPLIT = 200  # items 0-199 get 4 rows, items 200-399 get 2 rows

# Precompute indices
# For item i:
#   If i < SPLIT: use rows 0..3
#   If i >= SPLIT: use rows 0..1
_IDX = []
for item in range(400):
    d = D_HIGH if item < SPLIT else D_LOW
    row_indices = []
    for r in range(d):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _IDX.append(tuple(row_indices))


def update(state, item):
    buf = bytearray(state)
    idxs = _IDX[item]
    for r, idx in enumerate(idxs):
        off = r * W + idx
        v = buf[off]
        if v < 255:
            buf[off] = v + 1
    return bytes(buf)


def query(state, item):
    idxs = _IDX[item]
    m = 256
    for r, idx in enumerate(idxs):
        v = state[r * W + idx]
        if v < m:
            m = v
    return m
