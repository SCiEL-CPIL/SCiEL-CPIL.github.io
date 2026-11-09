"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Convert the vector to a single integer in base 3
    val = 0
    for v in vector:
        val = val * 3 + v
    
    # Apply a non-linear mixing function (MurmurHash3 finalizer style)
    # This breaks spatial correlations, helping the greedy algorithm
    # avoid local density traps and find a larger cap set.
    x = val
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    x = x ^ (x >> 31)
    
    # Mask to 64 bits to ensure consistent float conversion range
    x &= (1 << 64) - 1
    
    return float(x)
# EVOLVE-BLOCK-END
