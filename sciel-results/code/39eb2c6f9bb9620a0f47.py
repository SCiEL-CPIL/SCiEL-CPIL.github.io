import numpy as np
import time
import math
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using optimized NumPy sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Batch the additions for better cache performance
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [6, 11]

    # 2. Group by sigma value
    # Create an array of indices 1..MAX_N
    indices = np.arange(1, MAX_N + 1, dtype=np.int32)
    
    # Sort indices by their sigma value
    # We need to sort indices such that sigma[indices] is sorted
    # np.argsort returns indices that would sort the array
    sorted_indices = indices[np.argsort(sigma[indices], kind='mergesort')]
    
    if time.time() > deadline - 8:
        return [6, 11]

    # 3. Find best pair
    best_err = float('inf')
    best_pair = (6, 11)
    
    # Iterate through sorted indices to find groups of equal sigma
    n = len(sorted_indices)
    i = 0
    while i < n:
        if time.time() > deadline - 3:
            return [int(best_pair[0]), int(best_pair[1])]
            
        current_sigma = sigma[sorted_indices[i]]
        j = i + 1
        while j < n and sigma[sorted_indices[j]] == current_sigma:
            j += 1
            
        # Group is sorted_indices[i:j]
        if j - i >= 2:
            # Extract the group of numbers
            # Note: sorted_indices is sorted by sigma, not by value.
            # We need the group of values to be sorted by value for bisect.
            group = sorted_indices[i:j].astype(np.int64)
            group.sort()
            
            # Now search for best pair in this sorted group
            # For each b in group, find a in group closest to 1.5 * b
            # We can iterate through the group
            for k in range(len(group)):
                b = group[k]
                target = 1.5 * b
                
                # Find insertion point of target in group
                # group is a numpy array, we can use np.searchsorted
                pos = np.searchsorted(group, target, side='left')
                
                # Check neighbors pos and pos-1
                # We need to ensure we don't pick a=b
                candidates = []
                if pos < len(group):
                    candidates.append(pos)
                if pos > 0:
                    candidates.append(pos - 1)
                    
                for p in candidates:
                    a = group[p]
                    if a == b:
                        continue
                    
                    # Calculate error |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize this.
                    # diff = abs(2*a - 3*b)
                    # err = diff / (2.0 * b)
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue # Invalid, a/b = 1.5 exactly
                    err = diff / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (int(a), int(b))
                        
        i = j
        
    return [int(best_pair[0]), int(best_pair[1])]
