def construct():
    import numpy as np
    from math import gcd
    from functools import reduce
    import time

    deadline = time.time() + 80
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    vals = [L // d for d in denominators]
    
    mid = n // 2
    left_items = list(zip(vals[:mid], range(mid)))
    right_items = list(zip(vals[mid:], range(mid, n)))
    
    def gen_sums_with_idx(items):
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int32)
        for v, i in items:
            new_sums = sums + v
            new_idxs = idxs | (1 << i)
            sums = np.concatenate((sums, new_sums))
            idxs = np.concatenate((idxs, new_idxs))
        return sums, idxs

    s1, idx1 = gen_sums_with_idx(left_items)
    
    order1 = np.argsort(s1)
    s1_sorted = s1[order1]
    idx1_sorted = idx1[order1]
    
    # Free memory
    del s1, idx1, order1

    s2, idx2 = gen_sums_with_idx(right_items)
    
    targets = L - s2 - 1
    
    pos = np.searchsorted(s1_sorted, targets, side='right') - 1
    
    valid_mask = pos >= 0
    
    if not np.any(valid_mask):
        return [0]
        
    best_s1_vals = s1_sorted[pos[valid_mask]]
    best_idx1_vals = idx1_sorted[pos[valid_mask]]
    
    total_sums = best_s1_vals + s2[valid_mask]
    
    best_local_idx = np.argmax(total_sums)
    
    best_idx1 = best_idx1_vals[best_local_idx]
    best_idx2 = idx2[valid_mask][best_local_idx]
    
    selected_indices = []
    
    for k in range(mid):
        if best_idx1 & (1 << k):
            selected_indices.append(k)
            
    for k in range(len(right_items)):
        if best_idx2 & (1 << k):
            selected_indices.append(mid + k)
            
    if not selected_indices:
        selected_indices = [0]
        
    return selected_indices
