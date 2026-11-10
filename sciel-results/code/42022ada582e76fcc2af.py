"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1))

def _is_dissociated_ints(vals):
    """
    Check if all subset sums of the integer list 'vals' are distinct.
    """
    if not vals:
        return True
    sums = [0]
    for v in vals:
        # Append new sums
        # Using list comprehension is faster than loop in Python
        sums += [s + v for s in sums]
    # Check for duplicates
    # Using set is efficient for this size
    return len(set(sums)) == len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    # Universe is 1..22
    universe_size = 22
    # Calculate LCM of 1..22 to map reciprocals to integers
    lcm_val = _lcm_range(universe_size)
    
    # Map each i in 1..22 to LCM // i
    # index 0 corresponds to 1, index 1 to 2, etc.
    int_vals = [lcm_val // i for i in range(1, universe_size + 1)]
    
    # We want to find the largest subset of indices {0..21} such that
    # the corresponding int_vals form a dissociated set.
    
    best_set = []
    
    # Try sizes from 22 down to 1
    for k in range(universe_size, 0, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Iterate through all combinations of indices of size k
        # We can stop early if we find one
        for combo in itertools.combinations(range(universe_size), k):
            if time.time() > deadline - 2:
                break
                
            # Extract integer values for this combination
            vals = [int_vals[i] for i in combo]
            
            # Check if dissociated
            if _is_dissociated_ints(vals):
                # If we found a valid set of size k, it's the maximum
                # because we are iterating k downwards
                best_set = list(combo)
                found = True
                break
        
        if found:
            # Convert indices back to original values (1..22)
            # Index i corresponds to value i+1
            result = [i + 1 for i in best_set]
            return result
            
        # If we are running out of time and haven't found a set for this k,
        # we might want to try a greedy fallback for a smaller k?
        # But the loop will continue to k-1 which has fewer combinations.
        # If we run out of time in the inner loop, we break and go to next k.
        # If we run out of time completely, we return whatever best_set is (likely empty or from previous k if we stored it, but we break on found).
        # Actually, if we don't find any for k=22, best_set is empty.
        # If we find one for k=18, we return immediately.
        
    # Fallback if no set found within time (unlikely for small k)
    # Try a simple greedy set of small numbers?
    # The seed program returned [1, 2, 3, 5, 7, 11, 13, 17, 19] (size 9)
    # Let's just return a known valid small set if we fail completely
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
