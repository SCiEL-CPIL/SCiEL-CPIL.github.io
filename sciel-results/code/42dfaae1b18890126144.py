import struct
import time
import math

# Constants
NUM_ITEMS = 400
STATE_SIZE = 1024

# We will use a hybrid approach:
# 1. A Count-Min Sketch (CMS) with more columns (128) and 16-bit counters to reduce collision noise.
#    State layout: 4 rows * 128 cols * 2 bytes = 1024 bytes.
#    This uses the full state size but with smaller counters.
#    The seed used 4 rows * 64 cols * 4 bytes = 1024 bytes.
#    By increasing columns to 128, we halve the load factor per column, reducing the expected minimum error.

# Primes for hashing (same as seed to ensure good distribution)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647
NUM_ROWS = 4
NUM_COLS = 128

# Precompute hash indices for all 400 items for speed
# index[row][item] = ((item * p + 31 * row) % 2147483647) % 64  <-- wait, seed used % 64.
# If we use 128 columns, we should mod by 128.
# Let's check the seed code logic from the prompt:
# "index ((item * p + 31 * row) % 2147483647) % 64"
# If we change to 128 columns, we should use % 128 to utilize the space.

HASH_INDICES = [[0] * NUM_ITEMS for _ in range(NUM_ROWS)]
for row in range(NUM_ROWS):
    p = PRIMES[row]
    for item in range(NUM_ITEMS):
        idx = ((item * p + 31 * row) % MOD_LARGE) % NUM_COLS
        HASH_INDICES[row][item] = idx

def update(state, item):
    """
    Update the sketch state with a new item.
    """
    # Unpack 16-bit counters
    # 4 rows * 128 cols = 512 counters
    counters = list(struct.unpack('<512H', state))
    
    for row in range(NUM_ROWS):
        idx = HASH_INDICES[row][item]
        pos = row * NUM_COLS + idx
        counters[pos] += 1
        
    return struct.pack('<512H', *counters)

def query(state, item):
    """
    Query the frequency estimate for an item.
    """
    counters = struct.unpack('<512H', state)
    
    min_val = float('inf')
    for row in range(NUM_ROWS):
        idx = HASH_INDICES[row][item]
        pos = row * NUM_COLS + idx
        val = counters[pos]
        if val < min_val:
            min_val = val
            
    return min_val
