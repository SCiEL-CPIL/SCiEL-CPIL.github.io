"""
Optimized hybrid: exact counters for items 0-19 and 150-399 (270 items),
compact 3x37 CMS for sparse middle range 20-149 (130 items).

State layout (1024 bytes):
- Bytes 0-19:   20 uint8 counters for items 0-19 (exact)
- Bytes 20-149: 130 uint8 counters for items 150-279 mapped (exact)
  Actually: items 150-399 mapped to bytes 20-269 (250 items exact)
  Wait, 20 + 250 = 270 exact counters.
- Bytes 270-1023: 754 bytes for CMS (3 rows x 251 cols? No, let's use 3x250 = 750, leaving 4 spare)

Let me recalculate:
- Items 0-19: 20 exact counters (bytes 0-19)
- Items 150-399: 250 exact counters (bytes 20-269)
- Items 20-149: 130 items need CMS. Use 3 rows x 250 cols = 750 bytes (bytes 270-1019)
- Bytes 1020-1023: spare (4 bytes)

Total exact: 270 counters. CMS: 3x250 = 750 bytes. Total: 270+750 = 1020. Good.

Rationale:
- Stream 1 (uniform 0..399, 1024 items): 
  - Items 0-19: ~2.56 each, exact.
  - Items 150-399: ~2.56 each, exact (250 items).
  - Items 20-149: ~2.56 each, CMS (130 items). Total mass in CMS: 1024*130/400 = 332.8.
    Collision per cell: 332.8/250 ≈ 1.33. Min of 3 rows reduces error.
  
- Stream 2 (Zipf 1.6):
  - Most mass in items 0-19 (exact) and some in 150-399 (exact).
  - Items 20-149 have very low counts. CMS handles them well.
  
- Stream 3:
  - First 512: uniform 0-19, all exact.
  - Second 512: uniform 150-399, all exact (250 items).
  - Items 20-149: 0 occurrences. CMS cells stay 0.
"""

import struct

_NUM_EXACT_LOW = 20       # items 0-19
_NUM_EXACT_HIGH = 250     # items 150-399
_EXACT_LOW_OFFSET = 0
_EXACT_HIGH_OFFSET = 20   # bytes 20-269

_CMS_ROWS = 3
_CMS_COLS = 250
_CMS_OFFSET = 270

# Items 20-149 map to CMS
_CMS_ITEM_MIN = 20
_CMS_ITEM_MAX = 149

# Primes and seeds for 3 CMS rows
_PRIMES = [1035493289, 775374203, 1495862053]
_SEEDS = [31, 37, 41]
_MOD = 2147483647


def _cms_hash(item, row):
    """Hash item (20-149) to a column in the CMS."""
    p = _PRIMES[row]
    s = _SEEDS[row]
    return ((item * p + s) % _MOD) % _CMS_COLS


def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    
    if item < _NUM_EXACT_LOW:
        # Exact counter for items 0-19
        idx = _EXACT_LOW_OFFSET + item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    elif item >= 150:
        # Exact counter for items 150-399
        idx = _EXACT_HIGH_OFFSET + (item - 150)
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    else:
        # CMS for items 20-149
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = buf[offset]
            if val < 255:
                buf[offset] = val + 1
    
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    if item < _NUM_EXACT_LOW:
        return state[_EXACT_LOW_OFFSET + item]
    elif item >= 150:
        return state[_EXACT_HIGH_OFFSET + (item - 150)]
    else:
        # CMS: take min of 3 rows
        best = 255
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return best
