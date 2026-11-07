import struct
import time

# EVOLVE-BLOCK-START

# Structural change: Use a single 1024-byte state for a 16-row x 64-column
# Count-Min sketch with 8-bit counters (16 * 64 * 1 = 1024 bytes).
# This leverages more rows to reduce variance in the minimum estimator.
# We use optimized hash parameters to minimize collisions for the given distributions.

CM_ROWS = 16
CM_WIDTH = 64
MOD = 2147483647  # 2^31 - 1

# Hash constants: large odd primes/near-primes for good distribution
# The first 4 are from the seed; we add 12 more distinct constants
P_CONSTANTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    1618033989,
    1357246801,
    1111111111,
    999999937,
    888888887,
    777777773,
    666666661,
    555555557
]

def update(state, item):
    s = bytearray(state)
    
    for row in range(CM_ROWS):
        p = P_CONSTANTS[row]
        # Hash function: ((item * p + row * 31) % MOD) % CM_WIDTH
        idx = ((item * p + row * 31) % MOD) % CM_WIDTH
        pos = row * CM_WIDTH + idx
        
        # Read 1-byte counter, increment, cap at 255
        val = s[pos]
        if val < 255:
            s[pos] = val + 1
    
    return bytes(s)

def query(state, item):
    min_val = 256  # Start above max possible value
    
    for row in range(CM_ROWS):
        p = P_CONSTANTS[row]
        idx = ((item * p + row * 31) % MOD) % CM_WIDTH
        pos = row * CM_WIDTH + idx
        val = state[pos]
        if val < min_val:
            min_val = val
    
    return min_val

# EVOLVE-BLOCK-END
