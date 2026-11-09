import struct

# Constants
EXACT_COUNTS = 128  # Items 0-127 get exact uint8 counters
SKETCH_ITEMS_START = 128
SKETCH_ITEMS_END = 400
SKETCH_ITEM_COUNT = SKETCH_ITEMS_END - SKETCH_ITEMS_START  # 272 items

# Sketch parameters
SKETCH_ROWS = 4
SKETCH_COLS = 224  # 896 bytes / 4 rows = 224 bytes per row (uint8 counters)

# Hash primes for the sketch
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647

def _get_sketch_indices(item):
    """Get sketch indices for items >= SKETCH_ITEMS_START."""
    indices = []
    for row in range(SKETCH_ROWS):
        idx = ((item * PRIMES[row] + 31 * row) % MOD) % SKETCH_COLS
        indices.append(idx)
    return indices

def update(state, item):
    new_state = bytearray(state)
    
    if item < EXACT_COUNTS:
        # Exact counter: increment uint8 at position item
        current = new_state[item]
        if current < 255:
            new_state[item] = current + 1
    else:
        # Use sketch for items >= 128
        indices = _get_sketch_indices(item)
        for row in range(SKETCH_ROWS):
            idx = indices[row]
            # Offset: 128 (exact counters) + row * 224 + idx
            offset = EXACT_COUNTS + row * SKETCH_COLS + idx
            current = new_state[offset]
            if current < 255:
                new_state[offset] = current + 1
    
    return bytes(new_state)

def query(state, item):
    if item < EXACT_COUNTS:
        # Exact count
        return float(state[item])
    else:
        # Query sketch: return minimum across rows
        indices = _get_sketch_indices(item)
        min_val = 255
        for row in range(SKETCH_ROWS):
            idx = indices[row]
            offset = EXACT_COUNTS + row * SKETCH_COLS + idx
            val = state[offset]
            if val < min_val:
                min_val = val
        return float(min_val)
