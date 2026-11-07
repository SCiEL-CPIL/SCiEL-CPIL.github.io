"""
Pure Universal Count-Min Sketch + Bloom Filter.

Design:
- 8 rows of 64-bit counters (8 * 64 * 8 = 4096 bits = 512 bytes) at offset 0.
- 1024-bit Bloom filter (128 bytes) at offset 512.
- Total: 640 bytes. Remaining 384 bytes are zero padding.

Strategy:
- update: Increment the 8 CM counters for the item. Set the corresponding bits in the Bloom filter.
- query: 
  1. Check if the item is likely present using the Bloom filter.
  2. If NOT present, return 0.0 (avoids CM overestimation for absent items).
  3. If present, return the minimum of the 8 CM counter values.

This addresses the main source of error in CM sketches: collision noise causing non-zero estimates for items with true count 0.
"""

# EVOLVE-BLOCK-START

# Precompute hashes for speed
# We use 8 rows.
# Hash function: h(row, item) = ((item * P[row] + Q[row]) % 2**32) % 64
# We need to pick good constants P and Q.
# The seed used specific large primes. We can reuse or pick new ones.
# Let's use a standard set of large primes for better distribution.
P = [1035493289, 775374203, 1495862053, 2015145641, 1103515245, 1234567891, 987654321, 1357924680]
Q = [17, 31, 47, 61, 73, 89, 101, 113]

# Bloom filter parameters
BF_BITS = 1024
BF_NUM_HASHES = 3  # 3 hashes for 1024 bits and ~400 items is a good balance

def _bf_hash(item, salt):
    # Simple hash for Bloom filter bits
    # We want to map to 0..1023
    # Use a simple mixing function
    x = (item * 1000003 + salt) & 0xFFFFFFFF
    x ^= x >> 16
    x *= 0x45d9f3b
    x ^= x >> 16
    return x % BF_BITS

def update(state, item):
    state = bytearray(state)
    
    # 1. Update CM Sketch (8 rows, 64 counters each, 64-bit unsigned int)
    # Counters are stored as little-endian uint64 at offsets:
    # Row r: offset = r * 64 * 8 = r * 512
    # Counter c in row r: offset = r * 512 + c * 8
    
    for r in range(8):
        # Calculate index in 0..63
        idx = ((item * P[r] + Q[r]) % (2**32)) % 64
        offset = r * 512 + idx * 8
        
        # Read current value
        val = int.from_bytes(state[offset:offset+8], 'little', signed=False)
        # Increment
        val += 1
        # Write back (cap at max uint64, though unlikely to overflow)
        if val > (2**64 - 1):
            val = 2**64 - 1
        state[offset:offset+8] = val.to_bytes(8, 'little')
        
    # 2. Update Bloom Filter
    # Bloom filter is at offset 512, size 128 bytes (1024 bits)
    bf_offset = 512
    for i in range(BF_NUM_HASHES):
        bit_idx = _bf_hash(item, i)
        byte_idx = bf_offset + (bit_idx // 8)
        bit_mask = 1 << (bit_idx % 8)
        state[byte_idx] |= bit_mask
        
    return bytes(state)

def query(state, item):
    # 1. Check Bloom Filter
    bf_offset = 512
    present = True
    for i in range(BF_NUM_HASHES):
        bit_idx = _bf_hash(item, i)
        byte_idx = bf_offset + (bit_idx // 8)
        bit_mask = 1 << (bit_idx % 8)
        if not (state[byte_idx] & bit_mask):
            present = False
            break
            
    if not present:
        return 0.0
        
    # 2. Query CM Sketch
    min_val = 2**64 - 1
    for r in range(8):
        idx = ((item * P[r] + Q[r]) % (2**32)) % 64
        offset = r * 512 + idx * 8
        val = int.from_bytes(state[offset:offset+8], 'little', signed=False)
        if val < min_val:
            min_val = val
            
    return float(min_val)

# EVOLVE-BLOCK-END
