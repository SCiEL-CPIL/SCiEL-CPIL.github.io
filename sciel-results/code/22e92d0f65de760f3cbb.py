"""
Hybrid Count-Min Sketch with 8 rows x 64 columns of uint16 counters.
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

# Pre-compute indices for all possible items (0..399)
# _INDICES[item] is a tuple of 8 column indices
_INDICES = []
for item in range(400):
    cols = []
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        col = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        cols.append(col)
    _INDICES.append(tuple(cols))

def update(state, item):
    buf = bytearray(state)
    cols = _INDICES[item]
    for row in range(_NUM_ROWS):
        col = cols[row]
        # Byte offset for this counter (uint16 LE)
        off = (row * _NUM_COLS + col) * 2
        # Read current value
        val = buf[off] | (buf[off + 1] << 8)
        # Increment
        val += 1
        # Write back
        buf[off] = val & 0xFF
        buf[off + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    cols = _INDICES[item]
    min_val = 65535  # Max uint16
    for row in range(_NUM_ROWS):
        col = cols[row]
        off = (row * _NUM_COLS + col) * 2
        # Read from state (bytes)
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
