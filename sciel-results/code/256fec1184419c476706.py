import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Convert vector to a single integer in base 3
    # This is O(n) and fast for small n
    val = 0
    for x in reversed(vector):
        val = val * 3 + x
        
    # 64-bit MurmurHash3 finalizer for high-quality mixing
    # This breaks spatial correlations in the base-3 lattice
    val ^= val >> 33
    val = (val * 0xff51afd7ed558ccd) % (2**64)
    val ^= val >> 33
    val = (val * 0xc4ceb9fe1a85ec53) % (2**64)
    val ^= val >> 33
    
    # Add a small structural bias: vectors closer to the center (all 1s)
    # are slightly prioritized lower? Or higher?
    # Let's try penalizing vectors that are "too pure" (all 0s or all 2s)
    # as they are more likely to form APs with each other in simple structures.
    # Actually, in F3^n, (0,0,..0), (1,1,..1), (2,2,..2) form an AP.
    # So we should avoid picking them all.
    # The hash already scatters them.
    # Let's add a tiny noise based on the sum of coordinates to break remaining ties.
    
    # Calculate a simple feature: sum of coordinates
    # This is O(n). We already iterated. Let's do it in the first loop.
    
    return float(val)

# EVOLVE-BLOCK-END
