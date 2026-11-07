import struct

# State Layout:
# 0..399: uint8 exact counters
# 400..1023: 8 rows x 39 cols of uint16 CM sketch counters
# 8 * 39 * 2 = 624 bytes. 400 + 624 = 1024.

NUM_EXACT = 400
NUM_ROWS = 8
NUM_COLS = 39
SKETCH_OFFSET = 400

# Primes and salts for hashing
# We want good distribution mod 39
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1604569099,
    1103515245,
    1234567891,
    987654321
]
SALTS = [
    31, 37, 41, 43, 47, 53, 59, 61
]

def _get_sketch_index(item, row):
    # Simple mix
    h = (item * PRIMES[row] + SALTS[row]) 
    return h % NUM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # 1. Update exact counter
    if ba[item] < 255:
        ba[item] += 1
        
    # 2. Update CM sketch
    for r in range(NUM_ROWS):
        idx = _get_sketch_index(item, r)
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        
        # Read uint16
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        
        # Write uint16
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    # Read exact counter
    exact_val = state[item]
    
    if exact_val < 255:
        return exact_val
        
    # Overflow: use CM sketch
    min_val = 0xFFFF
    for r in range(NUM_ROWS):
        idx = _get_sketch_index(item, r)
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        
        val = state[offset] | (state[offset + 1] << 8)
        if val < min_val:
            min_val = val
            
    return min_val
