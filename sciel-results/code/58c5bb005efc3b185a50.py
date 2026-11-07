"""Improved CM sketch using uint16 counters with more columns to reduce collision noise."""
import struct

# Hash parameters from the seed
_P = (1035493289, 775374203, 1495862053, 2015145641)
_NUM_ROWS = 4
_NUM_COLS = 128
_MOD = 2147483647

def _index(item, row):
    p = _P[row]
    return ((item * p + 31 * row) % _MOD) % _NUM_COLS

def update(state, item):
    # Parse state as 4 rows of 128 uint16 values
    counters = list(struct.unpack('<256H', state))
    for row in range(_NUM_ROWS):
        idx = _index(item, row)
        pos = row * _NUM_COLS + idx
        counters[pos] = min(counters[pos] + 1, 65535)
    return struct.pack('<256H', *counters)

def query(state, item):
    counters = struct.unpack('<256H', state)
    min_val = 65536
    for row in range(_NUM_ROWS):
        idx = _index(item, row)
        pos = row * _NUM_COLS + idx
        if counters[pos] < min_val:
            min_val = counters[pos]
    return min_val
