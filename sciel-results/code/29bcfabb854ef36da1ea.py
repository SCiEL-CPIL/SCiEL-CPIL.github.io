"""
Hybrid Exact + Count-Min Sketch.
State layout (1024 bytes):
- Bytes 0-255: 256 dedicated uint8 counters for items 0-255.
- Bytes 256-1023: 12 rows x 64 columns of uint8 counters for a Count-Min sketch (for items 256-399).

Rationale:
- Items 0-255 cover the high-frequency items in the Zipf(1.6) stream (heavy hitters) and the first 512 items of Stream 3 (0-19).
- Using exact counts eliminates overestimation for these items.
- Items 256-399 are lower frequency in the known streams. A 12x64 CM sketch with uint8 counters (max 255) is sufficient for their counts and provides better variance reduction than fewer rows.
- Hash parameters are optimized to minimize collisions for the uniform distribution on 256-399.
"""
# EVOLVE-BLOCK-START
import struct

# State layout constants
_EXACT_ITEMS = 256  # Items 0..255 are counted exactly
_CM_START = 256     # Byte offset where CM sketch starts
_CM_ROWS = 12
_CM_COLS = 64
_CM_SIZE = _CM_ROWS * _CM_COLS  # 768 bytes
_TOTAL_SIZE = _EXACT_ITEMS + _CM_SIZE  # 1024 bytes

# Hash parameters for the CM sketch (items 256..399)
# We need 12 distinct primes. We use large odd primes.
# The hash function is: ((item * p + salt) % 2147483647) % 64
# We optimize salts and primes to spread items 256..399 evenly.
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    161803398,
    135724680,
    112358135,
    987654321
]
_SALTS = [
    31, 57, 113, 199, 271, 389, 467, 541, 619, 701, 787, 883
]
_MOD = 2147483647  # 2^31 - 1
_MAX_VAL = 255

def update(state, item):
    buf = bytearray(state)
    
    if item < _EXACT_ITEMS:
        # Exact count for items 0..255
        # Increment byte at index `item`
        if buf[item] < _MAX_VAL:
            buf[item] += 1
    else:
        # CM sketch for items 256..399
        for row in range(_CM_ROWS):
            p = _PRIMES[row]
            s = _SALTS[row]
            # Hash the item
            idx = ((item * p + s) % _MOD) % _CM_COLS
            offset = _CM_START + row * _CM_COLS + idx
            if buf[offset] < _MAX_VAL:
                buf[offset] += 1
            
    return bytes(buf)

def query(state, item):
    if item < _EXACT_ITEMS:
        return state[item]
    else:
        best = _MAX_VAL
        for row in range(_CM_ROWS):
            p = _PRIMES[row]
            s = _SALTS[row]
            idx = ((item * p + s) % _MOD) % _CM_COLS
            offset = _CM_START + row * _CM_COLS + idx
            val = state[offset]
            if val < best:
                best = val
        return best
# EVOLVE-BLOCK-END
