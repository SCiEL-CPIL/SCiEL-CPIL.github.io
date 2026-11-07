import struct

# Constants for CM Sketch
ROWS = 4
COLS = 60
# Primes for hashing
P = [1035493289, 775374203, 1495862053, 2015145641]
MOD = 2147483647

def update(state, item):
    new_state = bytearray(state)
    
    # 1. Update Bitset (Bytes 0-63)
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    new_state[byte_idx] |= mask
    
    # 2. Update CM Sketch (Bytes 64-1023)
    # Layout: Row r, Col c is at 64 + (r * COLS + c) * 4
    for r in range(ROWS):
        h = (item * P[r] + 31 * r) % MOD
        c = h % COLS
        offset = 64 + (r * COLS + c) * 4
        current_val = struct.unpack_from('<I', new_state, offset)[0]
        struct.pack_into('<I', new_state, offset, current_val + 1)
        
    return bytes(new_state)

def query(state, item):
    # 1. Check Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0.0
        
    # 2. Query CM Sketch
    min_val = float('inf')
    for r in range(ROWS):
        h = (item * P[r] + 31 * r) % MOD
        c = h % COLS
        offset = 64 + (r * COLS + c) * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
            
    return float(min_val)
