"""
Count-Min Sketch: 8 rows x 32 columns of uint16 counters.
State: 1024 bytes (256 uint16 values, little-endian).
Query: minimum across all 8 rows.
"""
import struct

ROWS = 8
COLS = 32
# Primes for each row's hash function
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    165411643,
    1063638847,
    336748561,
    2246822519,
]
# Secondary constants for diversity
Q_CONSTS = [31, 17, 43, 59, 23, 67, 71, 83]
M = 2147483647  # 2^31 - 1


def _hash(item, row):
    return ((item * PRIMES[row] + Q_CONSTS[row]) % M) % COLS


def update(state, item):
    new_state = bytearray(state)
    for row in range(ROWS):
        idx = _hash(item, row)
        pos = (row * COLS + idx) * 2  # byte offset for uint16
        val = struct.unpack_from('<H', new_state, pos)[0]
        new_val = val + 1
        if new_val > 65535:
            new_val = 65535
        struct.pack_into('<H', new_state, pos, new_val)
    return bytes(new_state)


def query(state, item):
    min_val = 65535
    for row in range(ROWS):
        idx = _hash(item, row)
        pos = (row * COLS + idx) * 2
        val = struct.unpack_from('<H', state, pos)[0]
        if val < min_val:
            min_val = val
    return min_val
