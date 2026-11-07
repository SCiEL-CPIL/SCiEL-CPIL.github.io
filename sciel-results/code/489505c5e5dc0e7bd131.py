"""4x64 32-bit Count-Min sketch with optimized hash parameters."""
# EVOLVE-BLOCK-START

import struct

_NUM_ROWS = 4
_NUM_COLS = 64
_M = (1 << 61) - 1  # Mersenne prime for modular reduction

# Hash parameters (a, b) for each row.
# Using distinct large primes and varied salts to decorrelate rows.
_HASH_PARAMS = [
    (1035493289, 0),
    (775374203, 1),
    (1495862053, 2),
    (2015145641, 3),
]

# Precompute column indices for O(1) access.
# _COLS[r][x] = column index for row r, item x.
_COLS = []
for r in range(_NUM_ROWS):
    a, b = _HASH_PARAMS[r]
    row = []
    for x in range(400):
        row.append(((x * a + b) % _M) % _NUM_COLS)
    _COLS.append(row)


def update(state, item):
    """Increment the 4 counters addressed by item. Returns new state bytes."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    s = bytearray(state)
    if len(s) < 1024:
        s = s + bytes(1024 - len(s))
    elif len(s) > 1024:
        s = s[:1024]
    
    for r in range(_NUM_ROWS):
        col = _COLS[r][item]
        offset = r * 256 + col * 4
        # Read current value
        val = struct.unpack_from('<I', s, offset)[0]
        val += 1
        # Write back (32-bit unsigned)
        struct.pack_into('<I', s, offset, val)
        
    return bytes(s)


def query(state, item):
    """Return the minimum counter value across the 4 rows for item."""
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < 1024:
        state = state + bytes(1024 - len(state))
        
    best = 0xFFFFFFFF
    for r in range(_NUM_ROWS):
        col = _COLS[r][item]
        offset = r * 256 + col * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < best:
            best = val
            
    return int(best)

# EVOLVE-BLOCK-END
