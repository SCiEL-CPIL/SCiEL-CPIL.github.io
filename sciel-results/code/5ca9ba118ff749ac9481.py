import struct

# Pre-compute CMS indices
CM_ROWS = 4
CM_COLS = 64
CM_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
CM_MOD_LARGE = 2147483647

# Layout
EXACT_16_BITS_COUNT = 56
EXACT_16_BITS_OFFSET = 0
EXACT_8_BITS_OFFSET = 112
CM_OFFSET = 456

# Pre-compute column indices for each item and row
# cm_indices[item][row] = column index
_cm_indices = [[0]*4 for _ in range(400)]
for item in range(400):
    for row in range(4):
        p = CM_PRIMES[row]
        _cm_indices[item][row] = ((item * p + 31 * row) % CM_MOD_LARGE) % CM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update exact counters
    if item < EXACT_16_BITS_COUNT:
        off = EXACT_16_BITS_OFFSET + item * 2
        val = ba[off] | (ba[off+1] << 8)
        if val < 65535:
            val += 1
        ba[off] = val & 0xFF
        ba[off+1] = (val >> 8) & 0xFF
    else:
        off = EXACT_8_BITS_OFFSET + (item - EXACT_16_BITS_COUNT)
        if ba[off] < 255:
            ba[off] += 1
            
    # Update CMS
    # We update all 4 rows
    cols = _cm_indices[item]
    for row in range(4):
        col = cols[row]
        off = CM_OFFSET + row * (CM_COLS * 2) + col * 2
        val = ba[off] | (ba[off+1] << 8)
        if val < 65535:
            val += 1
        ba[off] = val & 0xFF
        ba[off+1] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    # Get exact count
    if item < EXACT_16_BITS_COUNT:
        off = EXACT_16_BITS_OFFSET + item * 2
        exact_count = state[off] | (state[off+1] << 8)
        return float(exact_count)
    else:
        off = EXACT_8_BITS_OFFSET + (item - EXACT_16_BITS_COUNT)
        exact_count = state[off]
        
    if exact_count < 255:
        return float(exact_count)
    
    # Fallback to CMS
    cm_estimates = []
    cols = _cm_indices[item]
    for row in range(4):
        col = cols[row]
        off = CM_OFFSET + row * (CM_COLS * 2) + col * 2
        val = state[off] | (state[off+1] << 8)
        cm_estimates.append(val)
        
    return float(min(cm_estimates))
