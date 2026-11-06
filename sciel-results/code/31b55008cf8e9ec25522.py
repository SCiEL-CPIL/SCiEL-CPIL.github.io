import sys
import time

def priority(vector, n):
    # Calculate a deterministic pseudo-random priority for the vector.
    # This breaks the lexicographic ordering, which is known to be suboptimal
    # for greedy cap set construction. A randomized order typically yields
    # larger cap sets.
    
    # Convert vector to a single integer index
    idx = 0
    base = 1
    for i in range(n):
        idx += vector[i] * base
        base *= 3
        
    # Apply a bit-mixing function to ensure good distribution
    # Using a simple multiplicative hash followed by xor-shift
    h = idx
    h = (h ^ (h >> 16)) * 0x45d9f3b
    h = (h ^ (h >> 16)) * 0x45d9f3b
    h = h ^ (h >> 16)
    
    # Return as float for the interface
    return float(h)
