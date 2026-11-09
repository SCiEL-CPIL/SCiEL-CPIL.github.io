"""Program for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def _is_valid_numpy(A):
    """Check if all unordered triple sums (with replacement) are unique using NumPy."""
    n = len(A)
    if n < 3:
        return True
    
    A = np.array(A)
    
    # For small n, Python set might be faster due to overhead?
    # n=40 -> 64,000 sums. NumPy is likely faster.
    # Let's stick to NumPy as per directive.
    
    # Generate indices i, j, k such that 0 <= i <= j <= k < n
    # To avoid generating n^3 and masking, we can use combinations_with_replacement from itertools
    # which is in C? No, itertools is C but returns Python ints.
    # NumPy vectorization is better for "collision check".
    
    # Method:
    # i = [0,0,...,0, 1,1,...,1, ...]
    # j = [0,1,...,n-1, 0,1,...,n-1, ...]
    # k = [0,1,...,n-1, 0,1,...,n-1, ...]
    # This generates all n^3 tuples.
    # Then filter i<=j<=k.
    
    # For n=50, n^3 = 125,000. This is very fast in NumPy.
    
    i = np.repeat(np.arange(n), n*n)
    j = np.tile(np.repeat(np.arange(n), n), n)
    k = np.tile(np.arange(n), n*n)
    
    # Filter for i <= j <= k
    mask = (i <= j) & (j <= k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    sums = A[i] + A[j] + A[k]
    
    # Check for duplicates
    # np.unique sorts, so O(M log M). M is num sums.
    # Alternatively, use a set if M is small?
    # M for n=40 is ~ 10,000.
    # len(set(sums.tolist())) might be faster than np.unique for small M?
    # Let's try np.unique first.
    
    return len(np.unique(sums)) == len(sums)

def _greedy_build_randomized(max_val=4096, target_size=50, num_restarts=5, seed=42):
    """
    Build a set using randomized greedy strategies.
    """
    best_set = []
    rng = random.Random(seed)
    
    # Strategy 1: Greedy from 1
    selected = []
    for x in range(1, max_val + 1):
        if len(selected) >= target_size:
            break
        new_arr = selected + [x]
        if _is_valid_numpy(new_arr):
            selected = new_arr
    if len(selected) > len(best_set):
        best_set = selected
    if len(best_set) >= target_size:
        return best_set

    # Strategy 2: Randomized Greedy
    # Shuffle the range and pick the first valid ones
    for _ in range(num_restarts):
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        selected = []
        for x in candidates:
            if len(selected) >= target_size:
                break
            new_arr = selected + [x]
            if _is_valid_numpy(new_arr):
                selected = new_arr
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) >= target_size:
                break
                
    return best_set

def construct():
    """Construct a set of up to 80 distinct positive integers <= 4096 with unique triple sums."""
    # Theoretical max is ~41. We aim for that.
    # Use a target slightly higher to ensure we try hard, but cap at 80 as per interface.
    target = 45
    
    # First, try a standard greedy to get a baseline
    # This is fast with the NumPy check
    
    # We can also try to improve the result with a simple local search
    # But randomized greedy is often sufficient for this size.
    
    best_set = _greedy_build_randomized(max_val=4096, target_size=target, num_restarts=10, seed=123)
    
    return best_set
# EVOLVE-BLOCK-END
