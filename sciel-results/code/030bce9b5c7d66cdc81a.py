import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds margin to be safe within 90s limit
    
    N = 10**7
    
    # Compute sigma(n) for n=1..N using a sieve approach
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        if time.time() > deadline - 10:
            # Fallback if sieve is too slow
            return [11, 6]
        sigma[i::i] += i
    
    # We want to find pairs (a, b) such that sigma(a) == sigma(b) and |a/b - 3/2| is minimized.
    # This is equivalent to minimizing |2a - 3b| / (2b).
    # Let k = |2a - 3b|. We want to minimize k / (2b).
    # For a fixed k, we want to maximize b.
    
    # Strategy:
    # 1. Group indices by sigma value.
    # 2. For each group, find pairs (a, b) that minimize |2a - 3b| / (2b).
    
    # To do this efficiently, we can sort the indices in each group and then use a two-pointer
    # or binary search approach to find the best pair.
    
    # However, building groups for all 10^7 elements is expensive.
    # Instead, we can iterate through possible k values (1, 2, 3, ...) and for each k,
    # check if there exist a, b in the same sigma group such that 2a - 3b = k or 2a - 3b = -k.
    # This is also expensive.
    
    # Alternative strategy:
    # For each b, we want to find a such that sigma(a) == sigma(b) and a is close to 1.5 * b.
    # We can precompute a dictionary mapping sigma values to sorted lists of indices.
    # Then, for each b, we look up the list for sigma(b) and use binary search to find the
    # index closest to 1.5 * b.
    
    # Let's build the dictionary.
    # To save memory, we only store indices for sigma values that appear more than once.
    
    from collections import defaultdict
    
    # First, count occurrences of each sigma value
    sigma_counts = np.bincount(sigma, minlength=np.max(sigma) + 1)
    
    # Find sigma values that appear more than once
    repeated_sigmas = np.where(sigma_counts > 1)[0]
    
    # For each repeated sigma, collect the indices
    groups = {}
    for s in repeated_sigmas:
        if time.time() > deadline - 5:
            break
        indices = np.where(sigma == s)[0]
        if len(indices) > 1:
            groups[s] = indices
    
    # Now, for each b, find the best a in the same group
    best_a, best_b = 11, 6
    best_err = Fraction(abs(2*11 - 3*6), 2*6)  # |22 - 18| / 12 = 4/12 = 1/3
    
    # Iterate through b values
    # To speed up, we can iterate through b in a way that prioritizes larger b (since error is k/(2b))
    # But we need to check all b to be sure.
    
    # Let's iterate through b from N down to 1
    # For each b, get the group for sigma(b)
    # If the group exists, use binary search to find a close to 1.5 * b
    
    # To make this faster, we can pre-sort the groups (they are already sorted by np.where)
    
    import bisect
    
    for b in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
        
        s = sigma[b]
        if s not in groups:
            continue
        
        group = groups[s]
        target_a = 1.5 * b
        
        # Find position in group
        pos = bisect.bisect_left(group, target_a)
        
        # Check pos
        if pos < len(group):
            a = group[pos]
            if a != b:
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                err = Fraction(num, 2 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
        
        # Check pos - 1
        if pos - 1 >= 0:
            a = group[pos - 1]
            if a != b:
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                err = Fraction(num, 2 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
    
    return [int(best_a), int(best_b)]
