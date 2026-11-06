def priority(vector, n):
    # Encode the vector as a base-3 integer
    # This provides a unique ID for each vector in {0,1,2}^n
    val = 0
    for v in vector:
        val = val * 3 + v
    
    # Apply a mixing function to scramble the ID
    # This creates a pseudo-random permutation of the space
    # Using SplitMix64-style mixing
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Return as float
    # 53 bits of precision in float64 is sufficient to distinguish
    # all vectors for n up to ~18 (3^18 < 2^53)
    return float(val)
