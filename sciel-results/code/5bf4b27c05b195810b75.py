import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n <= N using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline:
        return [11, 6]  # Known valid pair: sigma(11)=12, sigma(6)=12, |11/6 - 1.5| = 1/6

    # 2. Bucket by sigma value using numpy argsort (fast)
    idx = np.arange(1, N + 1)
    sigma_vals = sigma[1:]
    order = np.argsort(sigma_vals, kind='stable')
    sorted_idx = idx[order]
    sorted_sigma = sigma_vals[order]
    
    # Find group boundaries
    changes = np.where(np.diff(sorted_sigma) != 0)[0] + 1
    group_starts = np.concatenate([[0], changes])
    group_ends = np.concatenate([changes, [len(sorted_idx)]])
    group_sizes = group_ends - group_starts
    
    if time.time() > deadline:
        # Fallback: find any valid pair
        for i in range(len(group_starts)):
            if group_sizes[i] >= 2:
                a = int(sorted_idx[group_starts[i]])
                b = int(sorted_idx[group_starts[i] + 1])
                if a != b:
                    return [a, b]
        return [11, 6]

    # 3. Search for best ratio within each group
    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this. Use exact integer comparison.
    
    best_num = 1  # numerator of error
    best_den = 6  # denominator of error, so error = 1/6 from pair (11,6)
    best_a = 11
    best_b = 6
    
    # Process groups in order of size (larger groups more likely to have good pairs)
    # But we need to be careful about time. Let's process all groups with size >= 2.
    # Optimize: for each group, use vectorized searchsorted.
    
    for i in range(len(group_starts)):
        if time.time() > deadline:
            break
        
        s = group_starts[i]
        e = group_ends[i]
        k = e - s
        if k < 2:
            continue
        
        group_vals = sorted_idx[s:e]  # already sorted by construction
        
        # For each b in group_vals, find a closest to 1.5*b
        b_vals = group_vals
        targets = (3 * b_vals) // 2  # integer division, approximate 1.5*b
        
        # Use searchsorted to find insertion points
        insert_pts = np.searchsorted(group_vals, targets, side='left')
        
        # Candidates: insert_pts and insert_pts - 1
        cand1 = np.clip(insert_pts, 0, k - 1)
        cand0 = np.clip(insert_pts - 1, 0, k - 1)
        
        a_cand1 = group_vals[cand1]
        a_cand0 = group_vals[cand0]
        
        # Compute error numerators: |2a - 3b|
        num1 = np.abs(2 * a_cand1 - 3 * b_vals)
        num0 = np.abs(2 * a_cand0 - 3 * b_vals)
        den = 2 * b_vals
        
        # Mask out a == b
        mask1 = (a_cand1 != b_vals)
        mask0 = (a_cand0 != b_vals)
        
        num1[~mask1] = np.iinfo(np.int64).max
        num0[~mask0] = np.iinfo(np.int64).max
        
        # Find best in this group
        min_num1 = np.min(num1)
        min_num0 = np.min(num0)
        
        if min_num1 < min_num0:
            min_num = min_num1
            b_idx = np.argmin(num1)
            a_val = a_cand1[b_idx]
            b_val = b_vals[b_idx]
        else:
            min_num = min_num0
            b_idx = np.argmin(num0)
            a_val = a_cand0[b_idx]
            b_val = b_vals[b_idx]
        
        if min_num == np.iinfo(np.int64).max:
            continue
        
        # Compare min_num / den[b_idx] with best_num / best_den
        # min_num / den < best_num / best_den  <=>  min_num * best_den < best_num * den
        den_val = den[b_idx]
        if min_num * best_den < best_num * den_val:
            best_num = int(min_num)
            best_den = int(den_val)
            best_a = int(a_val)
            best_b = int(b_val)
    
    # Ensure result is valid (not exact 3/2, and a != b)
    if best_num == 0:
        best_a, best_b = 11, 6
        best_num, best_den = 1, 6
    
    if best_a == best_b:
        best_a, best_b = 11, 6
    
    return [int(best_a), int(best_b)]
