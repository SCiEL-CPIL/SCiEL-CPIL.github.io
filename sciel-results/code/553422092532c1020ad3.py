"""
Count-Min Sketch with 8 rows x 64 columns of uint16 counters.
State is exactly 1024 bytes.
"""
# EVOLVE-BLOCK-START
import struct

# 8 distinct large odd primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
]
_MOD = 2147483647  # 2^31 - 1
_NUM_ROWS = 8
_NUM_COLS = 64
_COL_BITS = 16
_MAX_VAL = (1 << _COL_BITS) - 1


def update(state, item):
    buf = bytearray(state)
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = row * _NUM_COLS + idx
        val = buf[offset] | (buf[offset + 1] << 8)
        if val < _MAX_VAL:
            val += 1
        lo = val & 0xFF
        hi = (val >> 8) & 0xFF
        buf[offset] = lo
        buf[offset + 1] = hi
    return bytes(buf)


def query(state, item):
    best = _MAX_VAL
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = row * _NUM_COLS + idx
        val = state[offset] | (state[offset + 1] << 8)
        if val < best:
            best = val
    return best
# EVOLVE-BLOCK-END
