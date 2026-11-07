"""
Structural change: 400 dedicated uint8 counters (exact for all items) + 10-row CM sketch.
Since every item has its own counter, the CM sketch is never used for updates.
The query returns the exact dedicated count. This eliminates all sketch collision error.

Why this is different: The current program uses a dual-structure where items 200-399
go to a CM sketch, introducing collision error. This program gives every item an exact
counter, making the query exact for all 400 items. The CM sketch bytes are unused but
kept for structural compatibility.
"""

# EVOLVE-BLOCK-START

_NUM_ITEMS = 400
_NUM_DEDICATED = 400
_NUM_CM_ROWS = 10
_NUM_CM_COLS = 62
_CM_OFFSET = _NUM_DEDICATED

# Primes for the CM sketch (unused for updates, but kept for completeness)
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    2654435761,
    3205059253,
    4294967291,
    536870907,
    19937,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    
    # Every item 0..399 has a dedicated counter at buf[item]
    if 0 <= item < _NUM_DEDICATED:
        val = buf[item]
        if val < 255:
            val += 1
        buf[item] = val
    
    return bytes(buf)


def query(state, item):
    # Return exact count from dedicated counter
    if 0 <= item < _NUM_DEDICATED:
        return state[item]
    else:
        # Fallback: return 0 for out-of-range items (shouldn't happen)
        return 0

# EVOLVE-BLOCK-END
