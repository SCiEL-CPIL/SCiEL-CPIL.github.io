"""
Hybrid Exact-Counter + Compact CM Sketch.
State layout (1024 bytes):
- Bytes 0-255: 256 dedicated uint8 counters for items 0-255 (exact counting).
- Bytes 256-1023: 12 rows x 64 columns of uint8 counters for a Count-Min sketch
  used for items 256-399.

Items 0-255 are counted exactly.
Items 256-399 use the CM sketch with the minimum of 12 rows.
This eliminates overestimation for high-frequency items (Zipf heavy hitters,
Stream 3 first half) while maintaining good variance for the tail.
"""
# EVOLVE-BLOCK-START

_NUM_EXACT = 256
_NUM_CM_ROWS = 12
_NUM_CM_COLS = 64
_CM_OFFSET = _NUM_EXACT

# 12 distinct primes for hashing in the CM sketch part
# Using primes that provide good distribution for items 256-399
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    16807,
    48271,
    69621,
    28174,
    97531,
    131071,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    if item < _NUM_EXACT:
        # Exact counting for items 0-255
        val = buf[item]
        if val < 255:
            val += 1
        buf[item] = val
    else:
        # CM sketch for items 256-399
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = buf[offset]
            if val < 255:
                val += 1
            buf[offset] = val
    return bytes(buf)


def query(state, item):
    if item < _NUM_EXACT:
        # Exact count for items 0-255
        return state[item]
    else:
        # CM sketch minimum for items 256-399
        best = 255
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            val = state[_CM_OFFSET + row * _NUM_CM_COLS + idx]
            if val < best:
                best = val
        return best

# EVOLVE-BLOCK-END
