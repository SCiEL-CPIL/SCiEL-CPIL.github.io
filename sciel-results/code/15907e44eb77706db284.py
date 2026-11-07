import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    HARD_STOP = t0 + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Additive sieve
    # To speed up, we can skip the time check inside the loop if we are confident.
    # But let's keep it safe.
    # Actually, 10^7 iterations is ~1-2s in CPython for simple loop.
    # The vector ops are fast.
    # Let's just run it.
    
    for i in range(1, N + 1):
        if time.time() > HARD_STOP - 20.0:
            # Fallback
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() > HARD_STOP - 10.0:
        return [11, 6]
        
    # 2. Group by sigma
    # Use a dictionary to map sigma -> list of indices
    # This is faster than argsort for grouping.
    
    # However, creating 10^7 lists in a dict might be memory heavy?
    # Number of distinct sigma values is ~ N / log N? No, sigma(n) ~ n.
    # Distinct values are ~ N.
    # So we have ~ N keys, each with small list.
    # This is inefficient.
    
    # Better: Use argsort to get groups, then process groups.
    # But we want to process groups with large max-element first.
    
    indices = np.arange(1, N + 1, dtype=np.int64)
    order = np.argsort(sigma[1:], kind='stable')
    sorted_indices = order + 1
    sorted_sigma = sigma[sorted_indices]
    
    if time.time() > HARD_STOP - 5.0:
        return [11, 6]
        
    # Find group boundaries
    changes = np.nonzero(sorted_sigma[1:] != sorted_sigma[:-1])[0]
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [len(sorted_sigma)]))
    
    # Create a list of (max_element, start, end) for each group
    # And sort this list by max_element descending
    
    groups_info = []
    for gi in range(len(starts)):
        s0 = starts[gi]
        e0 = ends[gi]
        if e0 - s0 < 2:
            continue
        # Get max element in this group
        # sorted_indices[s0:e0] are the values
        # We need the max value
        # Since sorted_indices is not sorted by value, we must find max.
        # This is O(size) per group. Total O(N).
        # We can do this vectorized?
        # max_vals = np.max(sorted_indices[starts:ends], axis=1) ? No, slices are different lengths.
        
        # We can just compute max in the loop.
        # But we have ~ N groups? No, number of groups is number of distinct sigma values.
        # Which is ~ N.
        # So we have ~ N groups.
        # Looping N times is okay.
        
        # To speed up, we can just extract the slice and find max.
        # But doing this for N groups might be slow if many groups are small.
        # Most groups have size 1 or 2.
        # We skip size < 2.
        # So number of groups to process is much smaller.
        
        gn = sorted_indices[s0:e0]
        max_val = np.max(gn)
        groups_info.append((max_val, s0, e0, gn))
        
    # Sort groups by max_val descending
    groups_info.sort(key=lambda x: x[0], reverse=True)
    
    best_val = float('inf')
    best_pair = None
    
    # Iterate through groups
    for max_val, s0, e0, gn in groups_info:
        if time.time() > HARD_STOP - 2.0:
            break
            
        # Pruning: If 1.0 / (2.0 * max_val) >= best_val, 
        # then even with diff=1, we can't beat best_val.
        # Because min error for this group is approx 1/(2*max_val).
        # Actually, min error is min(|2a-3b|)/(2b).
        # Min |2a-3b| is 1 (if a!=b and 2a!=3b).
        # So min error >= 1/(2*max_val).
        # If 1/(2*max_val) >= best_val, skip.
        
        if 1.0 / (2.0 * max_val) >= best_val:
            continue
            
        # gn is already the array of values in the group.
        # We need to sort it to use bisect.
        # gn is a view? No, slice of numpy array.
        # We need a sorted list for bisect.
        # Converting to list and sorting is O(L log L).
        
        gn_sorted = gn.tolist()
        gn_sorted.sort()
        
        L = len(gn_sorted)
        
        for bi in range(L):
            b = gn_sorted[bi]
            target = 1.5 * b
            
            # Find a in gn_sorted closest to target
            j = bisect.bisect_left(gn_sorted, target)
            
            if j < L:
                a = gn_sorted[j]
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
            
            if j > 0:
                a = gn_sorted[j - 1]
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
                            
    if best_pair is None:
        return [11, 6]
        
    a, b = best_pair
    if sigma[a] != sigma[b]:
        return [11, 6]
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [int(a), int(b)]
