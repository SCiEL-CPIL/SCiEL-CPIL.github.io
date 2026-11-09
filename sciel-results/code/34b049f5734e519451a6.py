"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

# Precompute hashes for speed? 
# p in (1035493289, 775374203, 1495862053, 2015145641)
# Width 256, 4 rows.
# State is 1024 bytes.

def update(state, item):
    # state is bytes, returns bytes
    # Convert to bytearray for modification
    ba = bytearray(state)
    
    # Hash parameters
    p1 = 1035493289
    p2 = 775374203
    p3 = 1495862053
    p4 = 2015145641
    primes = [p1, p2, p3, p4]
    
    for r in range(4):
        p = primes[r]
        # Index calculation: ((item * p + 31 * row) % 2147483647) % 256
        idx = ((item * p + 31 * r) % 2147483647) % 256
        offset = r * 256 + idx
        
        # Increment, saturate at 255
        val = ba[offset] + 1
        if val > 255:
            val = 255
        ba[offset] = val
        
    return bytes(ba)

def query(state, item):
    # state is bytes
    p1 = 1035493289
    p2 = 775374203
    p3 = 1495862053
    p4 = 2015145641
    primes = [p1, p2, p3, p4]
    
    min_val = 255 # Max possible value
    
    for r in range(4):
        p = primes[r]
        idx = ((item * p + 31 * r) % 2147483647) % 256
        offset = r * 256 + idx
        val = state[offset]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
