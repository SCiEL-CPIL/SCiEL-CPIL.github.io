"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

# Structural change: Use all 1024 bytes for a 4-row CMS with 256 columns each.
# Rationale:
# - The current hybrid approach (128 exact + 4x224 CMS) scores 19.87.
# - Inspiration 2 (pure 4x256 CMS) scores 11.76.
# - The exact counters in the hybrid approach only cover items 0..127.
#   However, in the Zipf stream, items 128..399 still have significant counts
#   (albeit lower than 0..127), and in the uniform stream, all items are equal.
#   The collision noise in the 4x224 CMS for items >= 128 is higher than in 4x256.
# - By removing the exact counters and using the full 1024 bytes for the CMS,
#   we increase the width from 224 to 256 columns. This reduces the expected
#   collision noise (lambda) for all items, including those previously handled
#   exactly. While we lose exactness for items 0..127, the reduction in noise
#   for the remaining 272 items (and potentially better performance for the
#   exact items if their true counts are high enough that the CMS error is
#   small relative to their magnitude) should yield a lower overall MAE.
# - Specifically, for uniform streams, the error is dominated by collision noise.
#   Increasing width from 224 to 256 reduces the noise by a factor of ~256/224.
# - For Zipf, the heavy hitters (0..~50) have high counts, so CMS error is a
#   small fraction of the true count. The lighter items (50..399) benefit from
#   the wider sketch.
# - This is a structural change from "Hybrid Exact+CMS" to "Pure Wide CMS".

import struct

_ROWS = 4
_COLS = 256
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647

def _col(item, row):
    return ((item * _P[row] + 31 * row) % _MOD) % _COLS

def update(state, item):
    b = bytearray(state)
    for r in range(_ROWS):
        c = _col(item, r)
        off = r * _COLS + c
        b[off] = (b[off] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    m = 255
    for r in range(_ROWS):
        c = _col(item, r)
        v = state[r * _COLS + c]
        if v < m:
            m = v
    return m

# EVOLVE-BLOCK-END
