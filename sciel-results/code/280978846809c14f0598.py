"""Hybrid exact-counter + wider CMS sketch for 400-item universe."""
# EVOLVE-BLOCK-START
import struct

# Layout: bytes 0..255 = 128 x uint16 exact counters for items 0..127
#          bytes 256..1023 = 4 rows x 96 cols x uint16 CMS for items 128..399
#          (4*96*2 = 768 bytes)

EXACT_N = 128
CMS_ROWS = 4
CMS_COLS = 96
EXACT_BYTES = EXACT_N * 2  # 256
CMS_BYTES = CMS_ROWS * CMS_COLS * 2  # 768
TOTAL_BYTES = EXACT_BYTES + CMS_BYTES  # 1024

# Hash parameters for CMS (using different primes for each row)
# For item in 128..399, map to a 0..271 range, then hash into 96 buckets
# Use: index = ((item * p + q * row) % large_prime) % CMS_COLS
# We want good spread for 272 items into 96 buckets
_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
_LARGE_PRIME = 2147483647

def _cms_index(item, row):
    p = _PRIMES[row]
    # Offset item to avoid small values clustering; use item directly
    return ((item * p + 31 * (row + 1)) % _LARGE_PRIME) % CMS_COLS

def update(state, item):
    state = bytearray(state)
    if item < EXACT_N:
        # Exact counter
        off = item * 2
        val = struct.unpack_from('<H', state, off)[0]
        val += 1
        struct.pack_into('<H', state, off, val)
    else:
        # CMS update
        for row in range(CMS_ROWS):
            idx = _cms_index(item, row)
            off = EXACT_BYTES + (row * CMS_COLS + idx) * 2
            val = struct.unpack_from('<H', state, off)[0]
            val += 1
            struct.pack_into('<H', state, off, val)
    return bytes(state)

def query(state, item):
    if item < EXACT_N:
        off = item * 2
        return struct.unpack_from('<H', state, off)[0]
    else:
        # Return min over rows
        best = 0xFFFFFFFF
        for row in range(CMS_ROWS):
            idx = _cms_index(item, row)
            off = EXACT_BYTES + (row * CMS_COLS + idx) * 2
            val = struct.unpack_from('<H', state, off)[0]
            if val < best:
                best = val
        return best
# EVOLVE-BLOCK-END
