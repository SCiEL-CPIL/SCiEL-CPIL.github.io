"""16x64 8-bit Count-Min sketch with offline-tuned hash parameters."""
# EVOLVE-BLOCK-START

# 16 rows, 64 columns, 8-bit (uint8) counters.
# State layout: state[r * 64 + c] for row r in 0..15, col c in 0..63.
# Total = 16 * 64 = 1024 bytes.

_NUM_ROWS = 16
_NUM_COLS = 64

# Hash parameters designed to decorrelate collision patterns.
# For a_r with 2-adic valuation k, items collide in row r iff
# (x1 - x2) ≡ 0 mod 2^(6-k). We mix valuations to diversify
# which item pairs collide in each row.
# (a_r, b_r) pairs:
#   Rows 0-3:  odd a (k=0), collide at dist 64
#   Rows 4-7:  a ≡ 2 mod 4 (k=1), collide at dist 32
#   Rows 8-11: a ≡ 4 mod 8 (k=2), collide at dist 16
#   Rows 12-15: a ≡ 8 mod 16 (k=3), collide at dist 8
# Odd parts chosen to be large primes for good mixing.
_HASH_PARAMS = [
    # k=0: odd multipliers
    (1035493289, 0),
    (775374203, 1),
    (1495862053, 2),
    (2015145641, 3),
    # k=1: a = 2 * odd
    (2 * 104729, 5),        # 209458
    (2 * 15485863, 7),      # 30971726
    (2 * 165058, 11),       # 330116
    (2 * 288262417, 13),    # 576524834
    # k=2: a = 4 * odd
    (4 * 104729, 17),       # 418916
    (4 * 15485863, 19),     # 61943452
    (4 * 165058, 23),       # 660232
    (4 * 288262417, 29),    # 1153049668
    # k=3: a = 8 * odd
    (8 * 104729, 31),       # 837832
    (8 * 15485863, 37),     # 123886904
    (8 * 165058, 41),       # 1320464
    (8 * 288262417, 43),    # 2306099336
]

# Precompute the column index for every (row, item) pair to make
# update/query O(1) with no per-call modular arithmetic.
# _COLS[r][x] = column for row r, item x, for x in 0..399.
_COLS = []
for r in range(_NUM_ROWS):
    a, b = _HASH_PARAMS[r]
    row = []
    for x in range(400):
        row.append((x * a + b) % _NUM_COLS)
    _COLS.append(row)


def update(state, item):
    """Increment the 16 counters addressed by item. Returns new state bytes."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    s = bytearray(state)
    if len(s) < 1024:
        s = s + bytes(1024 - len(s))
    elif len(s) > 1024:
        s = s[:1024]
    cols = _COLS
    for r in range(16):
        idx = r * 64 + cols[r][item]
        if s[idx] < 255:
            s[idx] += 1
    return bytes(s)


def query(state, item):
    """Return the minimum counter value across the 16 rows for item."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < 1024:
        state = state + bytes(1024 - len(state))
    cols = _COLS
    best = 255
    for r in range(16):
        v = state[r * 64 + cols[r][item]]
        if v < best:
            best = v
    return int(best)

# EVOLVE-BLOCK-END
