"""Hybrid frequency estimator: exact 8-bit counters + Count-Min Sketch for overflow correction.

Layout: 1024 bytes total.
- Bytes 0..399: 8-bit exact counters for items 0..399.
  Since max count per stream is 1024, only items with count >= 256 overflow.
  In practice, at most 1-2 items per stream will hit this.
- Bytes 400..1023 (624 bytes): 8 rows x 39 cols x 16-bit CM sketch counters.
  8 * 39 * 2 = 624 bytes.

For query:
- If exact count < 255, return it (exact).
- If exact count == 255 (overflow), return the CM sketch estimate (min over rows),
  which overestimates but is much closer than 255 for true counts up to 1024.
"""

NUM_EXACT = 400
SKETCH_OFFSET = 400
NUM_ROWS = 8
NUM_COLS = 39

# Salt constants for CM sketch hashing (large odd numbers for good mixing)
SALT = (
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x9e3779b9,
    0x85ebca77,
    0xc2b2ae3d,
    0x27d4eb31,
)

def _hash_idx(item, row):
    """Bit-mixing hash to map (item, row) to a column index in 0..NUM_COLS-1."""
    h = (item + 1) * SALT[row]
    h &= 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update exact 8-bit counter (capped at 255)
    exact_val = ba[item]
    if exact_val < 255:
        ba[item] = exact_val + 1
    
    # Update CM sketch for all 8 rows
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    ba = state
    
    # Get exact 8-bit count
    exact_val = ba[item]
    
    if exact_val < 255:
        # Exact count, no overflow
        return exact_val
    
    # Overflow: use CM sketch estimate (min over rows, which overestimates true count)
    min_val = 0xFFFF
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
    
    return min_val
