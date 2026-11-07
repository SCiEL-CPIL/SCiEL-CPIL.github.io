"""
Optimized frequency sketch using exact 8-bit counters for all 400 items
and a wide 1-row Count-Min sketch for overflow correction.

Layout: 1024 bytes.
- Bytes 0..399: 8-bit exact counters for items 0..399.
- Bytes 400..1023: 312 16-bit counters (1 row x 312 cols).

Strategy:
- Exact counters handle all items with count < 255 perfectly.
- For the rare item with count >= 255 (Zipf peak), the wide sketch (W=312)
  provides a low-bias overestimate, minimizing the error compared to
  narrower sketches.
"""

NUM_EXACT = 400
NUM_COLS = 312
SKETCH_OFFSET = 400
MAX_EXACT = 255

# Salt for the single row hash
SALT = 0x9e3779b9

def _hash_idx(item):
    """Bit-mixing hash for the single row."""
    h = (item + 1) * SALT
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS

# Precompute index for each item
_IDX_TABLE = [_hash_idx(item) for item in range(400)]

def update(state, item):
    ba = bytearray(state)
    
    # Update exact counter
    if ba[item] < MAX_EXACT:
        ba[item] += 1
        
    # Update sketch
    idx = _IDX_TABLE[item]
    offset = SKETCH_OFFSET + idx * 2
    val = ba[offset] | (ba[offset + 1] << 8)
    val += 1
    if val > 0xFFFF:
        val = 0xFFFF
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    exact_val = state[item]
    if exact_val < MAX_EXACT:
        return exact_val
    
    # Overflow: use sketch
    idx = _IDX_TABLE[item]
    offset = SKETCH_OFFSET + idx * 2
    val = state[offset] | (state[offset + 1] << 8)
    return val
