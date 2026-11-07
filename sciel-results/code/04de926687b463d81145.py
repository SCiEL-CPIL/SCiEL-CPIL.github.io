"""Hybrid Exact + Count-Min Sketch for frequency estimation."""

# EVOLVE-BLOCK-START
import struct

# Primes for hashing
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    16777619,
    1299709
]
MOD = 2147483647
EXACT_COUNT = 128
CMS_ROWS = 6
CMS_COLS = 64
EXACT_BYTES = EXACT_COUNT * 2
CMS_BYTES = CMS_ROWS * CMS_COLS * 2
TOTAL_BYTES = EXACT_BYTES + CMS_BYTES # 256 + 768 = 1024

def update(state, item):
    state = bytearray(state)
    if item < EXACT_COUNT:
        # Update exact count
        offset = item * 2
        val = struct.unpack_from('<H', state, offset)[0]
        val += 1
        struct.pack_into('<H', state, offset, val)
    else:
        # Update CMS
        for r in range(CMS_ROWS):
            p = PRIMES[r]
            idx = ((item * p + 31 * r) % MOD) % CMS_COLS
            offset = EXACT_BYTES + (r * CMS_COLS + idx) * 2
            val = struct.unpack_from('<H', state, offset)[0]
            val += 1
            struct.pack_into('<H', state, offset, val)
    return bytes(state)

def query(state, item):
    if item < EXACT_COUNT:
        offset = item * 2
        return struct.unpack_from('<H', state, offset)[0]
    else:
        min_val = float('inf')
        for r in range(CMS_ROWS):
            p = PRIMES[r]
            idx = ((item * p + 31 * r) % MOD) % CMS_COLS
            offset = EXACT_BYTES + (r * CMS_COLS + idx) * 2
            val = struct.unpack_from('<H', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val
# EVOLVE-BLOCK-END
