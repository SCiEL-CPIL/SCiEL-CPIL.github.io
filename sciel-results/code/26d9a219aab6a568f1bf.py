import sys
import random
from itertools import combinations
from math import gcd
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    # Integer values v_i = L // i
    vals = [L // i for i in range(1, 23)]
    indices = list(range(1, 23))

    def is_dissociated(subset_indices):
        """
        Check if the subset of indices has distinct subset sums.
        """
        sums = {0}
        for idx in subset_indices:
            val = vals[idx - 1]
            # Optimization: if we have many sums, check collision early
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    # Try to find a large dissociated subset using randomized search
    # Start from high cardinality
    for k in range(22, 0, -1):
        # Limit iterations to respect time limit
        # For k=22, 2^22 ~ 4M sums. One check is slow.
        # For k=15, 2^15 ~ 32K sums. One check is fast.
        
        max_tries = 500 if k < 18 else 50
        
        for _ in range(max_tries):
            subset = random.sample(indices, k)
            if is_dissociated(subset):
                return subset
        
        # If we are getting low k, we can be more exhaustive or just accept
        # But random sampling might miss specific structures.
        # However, for "generic" dissociated sets, random works well.
        
        # Fallback for smaller k: deterministic check of all combinations?
        # No, too slow.
        
    # Fallback to the known good solution from the previous program
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
