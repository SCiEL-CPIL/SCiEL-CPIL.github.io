import numpy as np
import time
from bisect import bisect_left

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i

    if time.time() > deadline - 5.0:
        return [11, 6]

    # Build dictionary: sigma value -> list of n with that sigma
    # Use numpy to get unique sigma values and their indices
    # But this might use too much memory. Alternative: sort by sigma value.
    
    # Create array of (sigma[n], n) for n in 1..N, sort by sigma
    # This is O(N log N) which might be slow for N=10^7
    
    # Better: use a dictionary approach but only for sigma values that appear multiple times
    # First, find which sigma values appear more than once
    
    # Actually, let's use numpy to group efficiently
    # Sort indices by sigma value
    indices = np.arange(1, N + 1, dtype=np.int64)
    sorted_indices = indices[np.argsort(sigma[1:], kind='mergesort')]
    sorted_sigma = sigma[1:][sorted_indices]
    
    if time.time() > deadline - 3.0:
        return [11, 6]
    
    # Find groups where sigma value is the same
    # Use searchsorted to find boundaries
    unique_sigma, start_idx = np.unique(sorted_sigma, return_index=True)
    
    best_value = 1.0 / 3.0  # from [11, 6]
    best_pair = (11, 6)
    
    # For each group with size >= 2, search for best pair
    # Process groups in order of size (larger groups more likely to have good pairs)
    # But we need to be time-efficient
    
    # Get group sizes
    end_idx = np.empty(len(unique_sigma), dtype=np.int64)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = N
    group_sizes = end_idx - start_idx
    
    # Sort groups by size descending, process largest first
    group_order = np.argsort(-group_sizes)
    
    # Limit number of groups to process based on time
    max_groups_to_process = 200000
    
    for gi in group_order[:max_groups_to_process]:
        if time.time() > deadline - 2.0:
            break
        if group_sizes[gi] < 2:
            break  # sorted descending, so all remaining are < 2
        
        s_start = start_idx[gi]
        s_end = end_idx[gi]
        group_vals = sorted_indices[s_start:s_end]
        
        # For each b in group, find a closest to 3b/2 in group
        # Binary search in group_vals
        for bi in range(len(group_vals)):
            b = int(group_vals[bi])
            target = 3 * b / 2.0
            # Binary search for position where target would be
            pos = bisect_left(group_vals, int(target))
            # Check pos-1 and pos
            for check_pos in [pos - 1, pos]:
                if 0 <= check_pos < len(group_vals):
                    a = int(group_vals[check_pos])
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    val = num / (2.0 * b)
                    if val < best_value:
                        best_value = val
                        best_pair = (a, b)
    
    return [int(best_pair[0]), int(best_pair[1])]
