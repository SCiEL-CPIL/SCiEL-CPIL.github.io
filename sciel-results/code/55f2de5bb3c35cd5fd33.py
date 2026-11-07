import struct
import time

# State layout (1024 bytes total):
#   0..1023: 1024 x uint8 counters
#   Rows 0..15: 16 rows x 64 columns CM sketch (8-bit counters)
#   Total: 16 * 64 = 1024 counters

NUM_ROWS = 16
COUNTERS_PER_ROW = 64
TOTAL_COUNTERS = NUM_ROWS * COUNTERS_PER_ROW  # 1024

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z

def _get_index(item, row):
    # Mix item and row into a 64-bit hash, then mod by COUNTERS_PER_ROW
    seed = (item * 1035493289 + row * 775374203) & 0xFFFFFFFFFFFFFFFF
    return _splitmix64(seed) % COUNTERS_PER_ROW

# Pre-compute indices for all items 0..399 to speed up update/query
# _indices[item] = list of 16 absolute indices into the 1024-counter array
_INDICES = {}
for item in range(400):
    positions = []
    for row in range(NUM_ROWS):
        idx = _get_index(item, row)
        pos = row * COUNTERS_PER_ROW + idx
        positions.append(pos)
    _INDICES[item] = positions

def update(state, item):
    # Unpack all 1024 uint8 counters
    counters = list(state)  # state is bytes, so list(state) gives list of ints
    
    # Increment the 16 counters for this item
    for pos in _INDICES[item]:
        counters[pos] = min(counters[pos] + 1, 255)
    
    return bytes(counters)

def query(state, item):
    # state is bytes, so we can index directly
    # Return the minimum of the 16 counters
    min_val = 255
    for pos in _INDICES[item]:
        val = state[pos]
        if val < min_val:
            min_val = val
    return min_val
