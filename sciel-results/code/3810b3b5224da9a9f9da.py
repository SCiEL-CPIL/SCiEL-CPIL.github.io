import struct

# EVOLVE-BLOCK-START
def update(state, item):
    # State: 1024 bytes = 256 uint32 counters
    # We use 4 rows of 64 counters each.
    # To improve over the seed (which uses 4 rows of 64 uint32s but specific hashes),
    # we can use a different hashing scheme or more rows if possible?
    # No, state size is fixed at 1024 bytes.
    # Seed: 4 rows * 64 counters * 4 bytes/counter = 1024 bytes.
    # Alternative: 8 rows * 32 counters * 4 bytes/counter = 1024 bytes.
    # More rows reduces variance (min of more estimates).
    # Let's switch to 8 rows of 32 counters.
    
    s = bytearray(state)
    # Hash constants for 8 rows
    # Using simple distinct large primes or constants
    # We need to map item to index in 0..31 for each row.
    
    # To keep it simple and fast, we use a variation of the seed's hashing
    # but with modulus 32 instead of 64, and 8 rows.
    
    # Constants for 8 rows
    # Seed used: 1035493289, 775374203, 1495862053, 2015145641
    # Let's generate 4 more or use different ones.
    # Actually, let's just use the first 4 seed constants for rows 0-3 and
    # derive 4 more for rows 4-7 to ensure independence.
    
    P = [1035493289, 775374203, 1495862053, 2015145641, 1234567891, 987654321, 314159265, 271828183]
    
    for i in range(8):
        # Index calculation similar to seed but mod 32
        # Seed: ((item * p + 31 * row) % 2147483647) % 64
        # New: ((item * p + 31 * row) % 2147483647) % 32
        idx = ((item * P[i] + 31 * i) % 2147483647) % 32
        pos = i * 32 + idx
        
        # Extract uint32
        val = struct.unpack_from('<I', s, pos)[0]
        val += 1
        # Handle overflow (though unlikely for 1024 items)
        if val > 0xFFFFFFFF:
            val = 0
        struct.pack_into('<I', s, pos, val)
        
    return bytes(s)

def query(state, item):
    s = bytearray(state)
    P = [1035493289, 775374203, 1495862053, 2015145641, 1234567891, 987654321, 314159265, 271828183]
    
    min_val = float('inf')
    for i in range(8):
        idx = ((item * P[i] + 31 * i) % 2147483647) % 32
        pos = i * 32 + idx
        val = struct.unpack_from('<I', s, pos)[0]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
