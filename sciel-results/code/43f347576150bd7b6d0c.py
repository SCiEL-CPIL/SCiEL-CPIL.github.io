import struct
import time

# 16-row x 64-column Count-Min sketch with uint8 counters
# 16 * 64 = 1024 bytes total

_NUM_ROWS = 16
_NUM_COLS = 64

# Hash parameters: (prime, salt) for each row
# Chosen to provide good spread for items 0..399
_PARAMS = [
    (1035493289, 31),
    (775374203, 37),
    (1495862053, 41),
    (2015145641, 43),
    (1234567891, 47),
    (987654323, 53),
    (314159265, 59),
    (271828183, 61),
    (410332281, 67),
    (524287001, 71),
    (628318531, 73),
    (734003371, 79),
    (859915201, 83),
    (977019831, 89),
    (1097599973, 97),
    (1191792181, 101),
]

_MOD = 2147483647  # 2^31 - 1

# Pre-compute column indices for each item in each row
# col[item][row] = column index for item in row
_COLS = [[0] * _NUM_ROWS for _ in range(400)]
for item in range(400):
    for row in range(_NUM_ROWS):
        p, s = _PARAMS[row]
        _COLS[item][row] = ((item * p + s) % _MOD) % _NUM_COLS


def update(state, item):
    buf = bytearray(state)
    cols = _COLS[item]
    for row in range(_NUM_ROWS):
        col = cols[row]
        offset = row * _NUM_COLS + col
        val = buf[offset]
        if val < 0xFF:
            val += 1
        buf[offset] = val
    return bytes(buf)


def query(state, item):
    cols = _COLS[item]
    min_val = 0xFF
    for row in range(_NUM_ROWS):
        col = cols[row]
        offset = row * _NUM_COLS + col
        val = state[offset]
        if val < min_val:
            min_val = val
    return min_val
