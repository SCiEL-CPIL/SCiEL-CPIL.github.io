"""16x64 8-bit Count-Min sketch with scaling to handle heavy hitters."""
# EVOLVE-BLOCK-START

# 16 rows, 64 columns, 8-bit (uint8) counters.
# State layout: state[r * 64 + c] for row r in 0..15, col c in 0..63.
# Total = 16 * 64 = 1024 bytes.

_NUM_ROWS = 16
_NUM_COLS = 64
_M = (1 << 32) - 5  # Prime near 2^32 for better bit mixing
_SCALE = 4

# Hash parameters: (a, b)
# Using constants known for good mixing (odd multipliers).
_HASH_PARAMS = [
    (0x9E3779B9, 0),
    (0x85EBCA6B, 1),
    (0xC2B2AE35, 2),
    (0x27D4EB2F, 3),
    (0x165667B1, 5),
    (0x9E3779B1, 7),
    (0x85EBCA77, 11),
    (0xC2B2AE3D, 13),
    (0x27D4EB21, 17),
    (0x165667B9, 19),
    (0x9E3779A9, 23),
    (0x85EBCA5B, 29),
    (0xC2B2AE15, 31),
    (0x27D4EB4F, 37),
    (0x16566791, 41),
    (0x9E377999, 43),
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
    """Return the minimum counter value across the 16 rows for item, scaled."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < _NUM_ROWS * _NUM_COLS:
        state = state + bytes(_NUM_ROWS * _NUM_COLS - len(state))
    best = 255
    for r in range(_NUM_ROWS):
        v = state[r * _NUM_COLS + _COLS[r][item]]
        if v < best:
            best = v
    return int(best * _SCALE)

# EVOLVE-BLOCK-END
