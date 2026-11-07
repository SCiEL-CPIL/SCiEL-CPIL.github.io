"""
Hybrid frequency sketch: exact counters for known heavy-hitter ranges +
compact Count-Min sketch for the remaining items.

State layout (1024 bytes):
- Bytes 0-19:   20 uint8 exact counters for items 0-19 (Zipf heavy hitters)
- Bytes 20-269: 250 uint8 exact counters for items 150-399 (mapped to 20-269)
- Bytes 270-1023: 754 bytes for a 6-row x 125-col CMS with 8-bit counters
  for items 20-149 (130 items)

The three streams are:
1. Uniform on 0..399
2. Zipf(1.6) mod 400 (heavily biased toward low item IDs)
3. 512 uniform on 0..19 + 512 uniform on 150..399

Items 0-19 and 150-399 appear in streams 2 and 3 with high frequency.
Items 20-149 only appear in stream 1 (uniform), so they get ~2.56 expected
count per stream, low enough for a compact CMS to handle well.
"""

import struct

_EXACT_LOW_START = 0
_EXACT_LOW_END = 20  # items 0-19
_EXACT_HIGH_START = 150
_EXACT_HIGH_END = 400  # items 150-399

_CMS_ROWS = 6
_CMS_COLS = 125
_CMS_OFFSET = 270  # where CMS starts in state

# Hash parameters for the compact CMS (6 rows)
# Using distinct primes and seeds to minimize collision overlap
_PRIMES = [
    1035493289, 775374203, 1495862053,
    2015145641, 32452843, 2166136261
]
_SEEDS = [31, 37, 41, 43, 47, 53]
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
    
    if _EXACT_LOW_START <= item < _EXACT_LOW_END:
        # Exact counter for items 0-19
        offset = item
        if buf[offset] < 255:
            buf[offset] += 1
    elif _EXACT_HIGH_START <= item < _EXACT_HIGH_END:
        # Exact counter for items 150-399, mapped to bytes 20-269
        offset = 20 + (item - _EXACT_HIGH_START)
        if buf[offset] < 255:
            buf[offset] += 1
    else:
        # Items 20-149: use CMS
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            if buf[offset] < 255:
                buf[offset] += 1
    
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    if _EXACT_LOW_START <= item < _EXACT_LOW_END:
        # Exact count for items 0-19
        return state[item]
    elif _EXACT_HIGH_START <= item < _EXACT_HIGH_END:
        # Exact count for items 150-399
        return state[20 + (item - _EXACT_HIGH_START)]
    else:
        # Items 20-149: CMS minimum
        best = 255
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return best
