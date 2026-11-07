"""
Combined streaming frequency sketch:
- 400-bit membership bitset (50 bytes) to return 0 for unseen items.
- 4 rows of 128 uint16 counters (1024 - 50 = 974 bytes available; we use 1024 for counters, 
  so we need to fit within 974. 4*128*2 = 1024 > 974. 
  Let's use 4 rows of 122 uint16 counters = 4*122*2 = 976 bytes. Still > 974.
  4 rows of 121 uint16 counters = 4*121*2 = 968 bytes. 968 + 50 = 1018 <= 1024. Good.
  Or simply use 4 rows of 128 uint16 counters but overlap? No.
  
  Let's recalculate:
  Total budget: 1024 bytes.
  Bitset: 50 bytes (400 bits).
  Remaining: 974 bytes.
  We want 4 rows. Each row can have at most 974 // 4 // 2 = 121 counters.
  So 4 rows of 121 uint16 counters = 4 * 121 * 2 = 968 bytes.
  Total used: 50 + 968 = 1018 bytes. 6 bytes padding.
  
  Width w = 121. Depth d = 4.
  This is a structural change from the current program (256 buckets, 1 row) to a proper CM sketch (121 buckets, 4 rows).
  The bitset eliminates the error for unseen items (which is significant for uniform streams).
  The CM sketch with width 121 and depth 4 provides a better estimate for seen items than a single row of 256.
  
  Hashing:
  We need to map item (0..399) to 0..120 for each of the 4 rows.
  We can use the seed's hash structure but adapt to width 121.
  h = ((item * p + 31 * row) % 2147483647) % 121
  
  Primes: Use the seed's 4 primes.
"""

def update(state, item):
    b = bytearray(state)
    
    # Update bitset: set bit for item
    byte_idx = item // 8
    bit_idx = item % 8
    b[byte_idx] |= (1 << bit_idx)
    
    # Update 4 counters
    # Primes from seed
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    width = 121  # Number of counters per row
    counter_base = 50  # Counters start after the 50-byte bitset
    
    for row in range(4):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % width
        
        # Offset: row * width * 2 + idx * 2
        offset = counter_base + row * (width * 2) + idx * 2
        
        # Read uint16
        val = b[offset] | (b[offset + 1] << 8)
        val += 1
        # Write back
        b[offset] = val & 0xFF
        b[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    
    # Check bitset
    byte_idx = item // 8
    bit_idx = item % 8
    if not (b[byte_idx] & (1 << bit_idx)):
        return 0
    
    # Item is seen, return min of 4 counters
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    width = 121
    counter_base = 50
    
    min_val = 65535  # Max uint16
    
    for row in range(4):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % width
        
        offset = counter_base + row * (width * 2) + idx * 2
        
        val = b[offset] | (b[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
