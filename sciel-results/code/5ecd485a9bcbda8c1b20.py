import struct

# Hybrid: exact 8-bit counters for all 400 items + 2-row x 156-col uint16 CM sketch
# Layout: bytes 0..399 = exact counters, bytes 400..1023 = sketch (2*156*2=624 bytes)

NUM_EXACT = 400
NUM_ROWS = 2
NUM_COLS = 156
SKETCH_OFFSET = 400

# Salts for bit-mixing hash to map item -> column index in [0, 155]
SALT = (
    0x85ebca6b,
    0xc2b2ae35,
)

def _hash_idx(item, row):
    """Bit-mixing hash: multiply, XOR-shift, multiply, XOR-shift, mod 156."""
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
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
    
    # Update CM sketch (2 rows)
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
    
    # Overflow: use CM sketch estimate (which overestimates, so >= true count)
    min_val = 0xFFFF
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
    
    return min_val
