"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

# Configuration
NUM_ROWS = 8
NUM_COLS = 64
MOD = 2147483647

# Primes for hashing
P0 = 1035493289
P1 = 775374203
P2 = 1495862053
P3 = 2015145641
P4 = 1000000007
P5 = 1000000009
P6 = 1000000021
P7 = 1000000033
PRIMES = [P0, P1, P2, P3, P4, P5, P6, P7]

# Precompute row byte offsets
# Each row is 64 * 2 bytes = 128 bytes
ROW_OFFSETS = [r * 128 for r in range(NUM_ROWS)]

def update(state, item):
    s = bytearray(state)
    for row in range(NUM_ROWS):
        p = PRIMES[row]
        idx = ((item * p + row) % MOD) % NUM_COLS
        offset = ROW_OFFSETS[row] + idx * 2
        
        # Read uint16
        val = struct.unpack_from('<H', s, offset)[0]
        # Increment
        val += 1
        # Write back
        struct.pack_into('<H', s, offset, val)
        
    return bytes(s)

def query(state, item):
    min_val = 65535  # Max uint16
    for row in range(NUM_ROWS):
        p = PRIMES[row]
        idx = ((item * p + row) % MOD) % NUM_COLS
        offset = ROW_OFFSETS[row] + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
