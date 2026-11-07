import struct
import time

# Constants
NUM_ROWS = 4
COUNTERS_PER_ROW = 128
TOTAL_COUNTERS = NUM_ROWS * COUNTERS_PER_ROW
FORMAT = f"{TOTAL_COUNTERS}H"

# Pre-compute indices for all items and rows to speed up update/query
# Item range: 0..399
# Row range: 0..3
# Total entries: 400 * 4 = 1600 indices

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z

def _get_index(item, row):
    seed = (item * 1035493289 + row * 775374203) & 0xFFFFFFFFFFFFFFFF
    return _splitmix64(seed) % COUNTERS_PER_ROW

# Pre-compute the position in the flat array for each (item, row) pair
# pos = row * COUNTERS_PER_ROW + idx
_INDEX_TABLE = []
for item in range(400):
    row_indices = []
    for row in range(NUM_ROWS):
        idx = _get_index(item, row)
        pos = row * COUNTERS_PER_ROW + idx
        row_indices.append(pos)
    _INDEX_TABLE.append(row_indices)

def update(state, item):
    # Unpack all counters into a list of ints
    counters = list(struct.unpack(FORMAT, state))
    
    # Get the pre-computed positions for this item
    positions = _INDEX_TABLE[item]
    
    # Increment each counter
    for pos in positions:
        counters[pos] += 1
    
    return struct.pack(FORMAT, *counters)

def query(state, item):
    # Unpack all counters
    counters = struct.unpack(FORMAT, state)
    
    # Get the pre-computed positions for this item
    positions = _INDEX_TABLE[item]
    
    # Find the minimum value among the counters
    min_val = 65535
    for pos in positions:
        val = counters[pos]
        if val < min_val:
            min_val = val
            
    return min_val
