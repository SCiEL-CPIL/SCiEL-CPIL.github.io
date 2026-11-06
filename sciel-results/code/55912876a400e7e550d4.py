import math

def construct():
    # The maximum pairwise non-coprime set containing 255 is the set of all multiples of 3.
    # 255 = 3 * 5 * 17.
    # For square-free n, the max size is max_{p|n} floor(n/p).
    # |S_3| = 85, |S_5| = 51, |S_17| = 15.
    # 85 is the maximum.
    
    # Generate all multiples of 3 in [1, 255]
    # range(3, 256, 3) produces 3, 6, ..., 255
    result = list(range(3, 256, 3))
    
    return result
