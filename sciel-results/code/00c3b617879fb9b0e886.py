import struct
import time

# Constants
ROWS = 8
COLS = 64
# Multipliers for hashing
# We need 8 distinct multipliers.
# The seed used: 1035493289, 775374203, 1495862053, 2015145641
# Let's add 4 more large odd numbers.
MULTIPLIERS = [
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    2654435761,
    3221225473,
    3891068657,
    4246951201
]
SEEDS = [31, 61, 97, 151, 211, 281, 367, 461] # Small offsets

def _hash(item, row):
    # (item * p + seed) % 2^32 % 64
    # Using Python's arbitrary precision integers is fine, 
    # but we want to mimic 32-bit behavior for consistency if needed, 
    # though for modulo 64, lower bits are what matter.
    h = (item * MULTIPLIERS[row] + SEEDS[row]) & 0xFFFFFFFF
    return h % COLS

def update(state, item):
    # state is bytes of length 1024
    # 8 rows * 64 cols * 2 bytes/col = 1024
    # Convert to list of ints or array
    # For speed, we can use struct or array module, but simple list might be faster for 512 elements?
    # 1024 bytes is small.
    
    # Unpack 512 uint16s
    # struct.unpack is fast
    counters = list(struct.unpack('<512H', state))
    
    for r in range(ROWS):
        idx = _hash(item, r)
        # Index in the flat array: r * COLS + idx
        flat_idx = r * COLS + idx
        counters[flat_idx] += 1
        
    # Pack back
    new_state = struct.pack('<512H', *counters)
    return new_state

def query(state, item):
    counters = struct.unpack('<512H', state)
    
    min_val = float('inf')
    for r in range(ROWS):
        idx = _hash(item, r)
        flat_idx = r * COLS + idx
        val = counters[flat_idx]
        if val < min_val:
            min_val = val
            
    return min_val
