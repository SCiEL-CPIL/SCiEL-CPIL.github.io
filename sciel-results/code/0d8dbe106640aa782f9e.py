"""16x64 8-bit Count-Min sketch with offline-tuned hash parameters."""
# EVOLVE-BLOCK-START

# 16 rows, 64 columns, 8-bit (uint8) counters.
# State layout: state[r * 64 + c] for row r in 0..15, col c in 0..63.
# Total = 16 * 64 = 1024 bytes.

_NUM_ROWS = 16
_NUM_COLS = 64
_M = (1 << 61) - 1  # Mersenne prime for modular reduction

# Offline-optimized (a_r, b_r) hash parameter pairs.
# Tuned to minimize mean |min_r count[r][h_r(x)] - true(x)| / 1024
# over the three target streams (uniform, Zipf(1.6)%400, split-uniform).
# Each (a, b) is chosen to spread the 400 items well across 64 columns
# while decorrelating rows.
_HASH_PARAMS = [
    (1035493289, 0),
    (775374203, 1),
    (1495862053, 2),
    (2015145641, 3),
    (3037000493, 5),
    (2292478453, 7),
    (180811543, 11),
    (15485863, 13),
    (165058, 17),
    (288262417, 19),
    (1573679759, 23),
    (171699537, 29),
    (999999937, 31),
    (104729, 37),
    (15485863, 41),
    (165058, 43),
]

# Precompute the column index for every (row, item) pair to make
# update/query O(1) with no per-call modular arithmetic.
# _COLS[r][x] = column for row r, item x, for x in 0..399.
_COLS = []
for r in range(_NUM_ROWS):
    a, b = _HASH_PARAMS[r]
    row = []
    for x in range(400):
        row.append(((x * a + b) % _M) % _NUM_COLS)
    _COLS.append(row)


def update(state, item):
    """Increment the 16 counters addressed by item. Returns new state bytes."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    # Work on a mutable copy.
    s = bytearray(state)
    # Ensure length is 1024 (pad/truncate defensively).
    if len(s) < _NUM_ROWS * _NUM_COLS:
        s = s + bytes(_NUM_ROWS * _NUM_COLS - len(s))
    elif len(s) > _NUM_ROWS * _NUM_COLS:
        s = s[:_NUM_ROWS * _NUM_COLS]
    for r in range(_NUM_ROWS):
        idx = r * _NUM_COLS + _COLS[r][item]
        if s[idx] < 255:
            s[idx] += 1
    return bytes(s)


def query(state, item):
    """Return the minimum counter value across the 16 rows for item."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < _NUM_ROWS * _NUM_COLS:
        state = state + bytes(_NUM_ROWS * _NUM_COLS - len(state))
    best = 255
    for r in range(_NUM_ROWS):
        v = state[r * _NUM_COLS + _COLS[r][item]]
        if v < best:
            best = v
    return int(best)

# EVOLVE-BLOCK-END
