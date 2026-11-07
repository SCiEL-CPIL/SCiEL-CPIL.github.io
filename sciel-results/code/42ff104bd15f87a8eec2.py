"""Count-Min sketch with 16 rows x 64 uint8 counters, tuned hash parameters."""
# EVOLVE-BLOCK-START

import struct

_NUM_ROWS = 16
_NUM_COLS = 64

# Hash parameters: (a, b) for each row.
# Index = ((item * a + b) % 2147483647) % 64
# These are chosen to spread 400 items well across 64 columns with good decorrelation.
_HASH_PARAMS = [
    (1035493289, 0),
    (775374203, 31),
    (1495862053, 62),
    (2015145641, 93),
    (3037000493, 124),
    (2292478453, 155),
    (180811543, 186),
    (15485863, 217),
    (165058, 248),
    (288262417, 279),
    (4294967291, 310),
    (362437, 341),
    (998244353, 372),
    (179424673, 403),
    (6151, 434),
    (2015145641, 465),
]

_M = 2147483647  # 2^31 - 1, Mersenne prime


def update(state, item):
    """Increment counters in all 16 rows for the given item."""
    state = bytearray(state)
    for r in range(_NUM_ROWS):
        a, b = _HASH_PARAMS[r]
        idx = ((item * a + b) % _M) % _NUM_COLS
        pos = r * _NUM_COLS + idx
        state[pos] = min(state[pos] + 1, 255)
    return bytes(state)


def query(state, item):
    """Return the minimum counter value across all 16 rows as a frequency estimate."""
    min_val = 255
    for r in range(_NUM_ROWS):
        a, b = _HASH_PARAMS[r]
        idx = ((item * a + b) % _M) % _NUM_COLS
        pos = r * _NUM_COLS + idx
        v = state[pos]
        if v < min_val:
            min_val = v
    return min_val

# EVOLVE-BLOCK-END
