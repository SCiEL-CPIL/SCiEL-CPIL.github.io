"""
Pure 16-row × 64-col Count-Min sketch using all 1024 bytes.
Each row is an independent hash of the item; query returns the minimum.
This maximizes the number of independent estimates, reducing the probability
that a collision inflates the minimum.
"""
# EVOLVE-BLOCK-START

_NUM_ROWS = 16
_NUM_COLS = 64

# 16 distinct large primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    16807,
    48271,
    69621,
    28174,
    97531,
    131071,
    65537,
    32749,
    16381,
    8191,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = row * _NUM_COLS + idx
        val = buf[offset]
        if val < 255:
            val += 1
        buf[offset] = val
    return bytes(buf)


def query(state, item):
    best = 255
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        val = state[row * _NUM_COLS + idx]
        if val < best:
            best = val
    return best

# EVOLVE-BLOCK-END
