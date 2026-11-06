import math
import time

def construct():
    # The maximum pairwise non-coprime set containing 255 is the set of all multiples of 3.
    # 255 = 3 * 5 * 17.
    # Let S_p be the set of multiples of p in [1, 255].
    # |S_3| = 85, |S_5| = 51, |S_17| = 15.
    # It is a known result that for square-free n, the max size is max_{p|n} floor(n/p).
    # Thus, the set of multiples of 3 is optimal with size 85.
    
    result = []
    for i in range(3, 256, 3):
        result.append(i)
    
    # 255 is a multiple of 3, so it is included.
    # We ensure 255 is in the list just in case of any off-by-one errors in logic, 
    # though range(3, 256, 3) ends at 255.
    if 255 not in result:
        result.append(255)
        
    return result
