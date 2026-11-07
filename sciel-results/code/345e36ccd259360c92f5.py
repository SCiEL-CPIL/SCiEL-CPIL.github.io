"""Optimized Count-Min Sketch with 8 rows and good hashing.

Layout: 1024 bytes total.
- 8 rows x 64 counters x 2 bytes each (16-bit counters).
  Total capacity: 8 * 64 * 2^16 >> max possible count of 1024, so no overflow.
  8 * 64 * 2 = 1024 bytes.

Hashing: H_j(i) = ((i * p_j + b_j) % 2147483647) % 64
We use 8 distinct large primes for p_j to ensure good distribution.
"""

# Pre-compute hash constants
# Using primes or large odd numbers coprime to M=2147483647
P_CONSTS = (
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1000000007,
    1000000009,
    1000000021,
    1000000033,
)

# Offsets
B_CONSTS = (
    31,
    57,
    91,
    133,
    199,
    257,
    311,
    367,
)

NUM_ROWS = 8
NUM_COLS = 64
MOD_LARGE = 2147483647

def update(state, item):
    ba = bytearray(state)
    
    for j in range(NUM_ROWS):
        # Calculate index using large modulus to decorrelate
        idx = ((item * P_CONSTS[j] + B_CONSTS[j]) % MOD_LARGE) % NUM_COLS
        offset = (j * NUM_COLS + idx) * 2
        
        # Read 16-bit little-endian integer
        val = ba[offset] | (ba[offset + 1] << 8)
        
        # Increment
        val += 1
        
        # Write back
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    ba = state
    min_val = float('inf')
    
    for j in range(NUM_ROWS):
        idx = ((item * P_CONSTS[j] + B_CONSTS[j]) % MOD_LARGE) % NUM_COLS
        offset = (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
