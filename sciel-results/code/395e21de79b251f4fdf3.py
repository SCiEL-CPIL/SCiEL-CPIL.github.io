from functools import lru_cache
import time

def construct():
    # LCM of 1..24
    # 1..24 LCM is 5354228880
    LCM = 5354228880
    
    # Greedy construction from largest to smallest
    # This ensures that when we add x, we only need to check if 1/x is a sum of 1/y for y already in S (y > x).
    # We don't need to re-check existing elements because 1/x > 1/y for all y in S,
    # so any sum involving 1/x is > 1/x > 1/y, thus cannot equal 1/y.
    
    S = []
    subset_sums = {0}
    
    # We iterate from 24 down to 1
    for x in range(24, 0, -1):
        val = LCM // x
        if val not in subset_sums:
            S.append(x)
            # Update subset sums
            # To avoid modifying while iterating, create a new set
            new_sums = set(subset_sums)
            for s in subset_sums:
                new_sums.add(s + val)
            subset_sums = new_sums
            
    return S
