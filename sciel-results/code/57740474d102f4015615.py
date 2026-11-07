# Constants for the sketch
# Using 12-bit counters to allow more columns (170 vs 128) within 1024 bytes
# 4 rows * 170 cols * 12 bits = 8160 bits = 1020 bytes
NUM_ROWS = 4
NUM_COLS = 170
STATE_BYTES = 1024

# Primes for hashing
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647

def update(state, item):
    """
    Updates the frequency sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes
    """
    val = int.from_bytes(state, 'little')
    
    for row in range(NUM_ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % NUM_COLS
        pos = row * NUM_COLS + idx
        shift = pos * 12
        mask = 0xFFF << shift
        current = (val & mask) >> shift
        new_val = current + 1
        val = (val & ~mask) | (new_val << shift)
        
    return val.to_bytes(STATE_BYTES, 'little')

def query(state, item):
    """
    Queries the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: non-negative frequency estimate (int)
    """
    val = int.from_bytes(state, 'little')
    min_count = 4095 # Max 12-bit value
    
    for row in range(NUM_ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % NUM_COLS
        pos = row * NUM_COLS + idx
        shift = pos * 12
        count = (val >> shift) & 0xFFF
        if count < min_count:
            min_count = count
            
    return min_count
