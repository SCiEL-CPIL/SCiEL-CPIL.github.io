import struct

# EVOLVE-BLOCK-START

_ROWS = 8
_COLS = 64
_MOD = 2147483647

# 8 large odd constants for hashing
# Using primes or random large odd numbers
_P = (
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2654435761,
    3001150337,
    3221225473,
    3735928559
)

# Pre-compute indices for all 400 items
# _OFFS[item] is a tuple of 8 byte offsets (each offset points to a uint16 LE)
_OFFS = []
for item in range(400):
    offsets = []
    for row in range(_ROWS):
        p = _P[row]
        # Hash function
        idx = ((item * p + row * 1234567) % _MOD) % _COLS
        # Byte offset: row * 64 * 2 + idx * 2
        byte_off = (row * _COLS + idx) * 2
        offsets.append(byte_off)
    _OFFS.append(tuple(offsets))

def update(state, item):
    ba = bytearray(state)
    for off in _OFFS[item]:
        # Read uint16 LE
        val = ba[off] | (ba[off + 1] << 8)
        val += 1
        # Write uint16 LE
        ba[off] = val & 0xFF
        ba[off + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    min_val = 0xFFFF
    for off in _OFFS[item]:
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val
    return min_val

# EVOLVE-BLOCK-END
