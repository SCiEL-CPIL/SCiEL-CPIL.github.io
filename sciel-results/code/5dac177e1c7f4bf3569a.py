"""Self-contained Count-Min sketch (4x64) with direct byte updates/queries."""

import struct

# EVOLVE-BLOCK-START
ROWS = 4
WIDTH = 64
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647
MASK32 = 0xFFFFFFFF

def _row_col(item, row):
    p = PRIMES[row]
    return (item * p + 31 * row) % MOD % WIDTH

def update(state, item):
    """Increment the CM counters for item; return new state as bytes."""
    buf = bytearray(state)
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = (row * WIDTH + col) * 4
        val = struct.unpack_from("<I", buf, idx)[0]
        val = (val + 1) & MASK32
        struct.pack_into("<I", buf, idx, val)
    return bytes(buf)

def query(state, item):
    """Return min over rows of the CM estimate for item."""
    best = 0xFFFFFFFF
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = (row * WIDTH + col) * 4
        val = struct.unpack_from("<I", state, idx)[0]
        if val < best:
            best = val
    return best
# EVOLVE-BLOCK-END
