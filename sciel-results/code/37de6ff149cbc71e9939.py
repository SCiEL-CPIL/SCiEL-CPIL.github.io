"""
Logarithmic-bucketed Count-Min sketch.
State layout (1024 bytes):
  16 rows x 64 columns of uint8, each storing ceil(log2(count+1)).
  Update: popcount the row's 64 uint8s to get total, compute new bucket,
          set the target cell to max(old, new_bucket).
  Query: min across all 16 rows of (2^stored - 1).
  This gives a multiplicative error bound of <2x instead of additive CM error.
"""
# EVOLVE-BLOCK-START

import struct

_NUM_ROWS = 16
_NUM_COLS = 64
_TOTAL_CELLS = _NUM_ROWS * _NUM_COLS  # 1024

# 16 well-distributed odd primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    234567891,
    345678911,
    567890123,
    789012345,
    901234567,
    123456789,
    987654321,
    876543210,
    765432109,
    654321098,
]
_MOD = 2147483647


def _popcount_row(buf, row_start):
    """Sum of all 64 uint8 values in one row (total count across all items)."""
    total = 0
    for i in range(_NUM_COLS):
        total += buf[row_start + i]
    return total


def _bucket(count):
    """ceil(log2(count + 1)), capped at 255."""
    if count <= 0:
        return 0
    if count == 1:
        return 1
    # count >= 2: ceil(log2(count+1))
    # log2(2)=1 -> ceil=1, log2(3)=1.585 -> ceil=2, log2(4)=2 -> ceil=2,
    # log2(5)=2.32 -> ceil=3, log2(8)=3 -> ceil=3, log2(9)=3.17 -> ceil=4
    return count.bit_length()


def update(state, item):
    buf = bytearray(state)

    # Precompute all 16 row indices
    indices = []
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        indices.append(row * _NUM_COLS + idx)

    # Compute total count from any row (all rows have the same total)
    # Use row 0 for the popcount
    total = 0
    row0_start = 0
    for i in range(_NUM_COLS):
        total += buf[row0_start + i]

    # New bucket for this item's count
    new_bucket = _bucket(total + 1)

    # Update: set each cell to max(current, new_bucket)
    for abs_idx in indices:
        cur = buf[abs_idx]
        if new_bucket > cur:
            buf[abs_idx] = new_bucket

    return bytes(buf)


def query(state, item):
    best = 255
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        val = state[row * _NUM_COLS + idx]
        if val < best:
            best = val
    # Convert bucket back to count estimate: 2^bucket - 1
    return (1 << best) - 1

# EVOLVE-BLOCK-END
