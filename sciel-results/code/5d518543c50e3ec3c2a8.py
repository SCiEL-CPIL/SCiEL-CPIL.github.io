"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

# Structural change: Two-tier design with exact counters for items 0-255
# and a 4x192 CMS for items 256-399.
# Rationale:
# - In Zipf(1.6), items 0-255 capture ~99%+ of the mass. Exact counters give
#   zero error for these high-frequency items.
# - Items 256-399 have low true counts (typically < 5 in Zipf, ~1.28 in uniform).
#   A 4x192 CMS provides reasonable estimates with collision noise that is
#   small relative to the true counts for these lighter items.
# - Layout: bytes 0-255 = exact uint8 counters for items 0-255
#           bytes 256-1023 = 4 rows x 192 cols of uint8 CMS for items 256-399
# - This is a structural change from "Pure Wide CMS" to "Tiered Exact+CMS".

_ROWS = 4
_COLS = 192
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647
_EXACT_COUNT = 256

def _col(item, row):
    # Map item (256-399) to column 0-191
    return ((item * _P[row] + 31 * row) % _MOD) % _COLS

def update(state, item):
    b = bytearray(state)
    if item < _EXACT_COUNT:
        b[item] = (b[item] + 1) & 0xFF
    else:
        for r in range(_ROWS):
            c = _col(item, r)
            off = _EXACT_COUNT + r * _COLS + c
            b[off] = (b[off] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    if item < _EXACT_COUNT:
        return state[item]
    else:
        m = 255
        for r in range(_ROWS):
            c = _col(item, r)
            v = state[_EXACT_COUNT + r * _COLS + c]
            if v < m:
                m = v
        return m

# EVOLVE-BLOCK-END
