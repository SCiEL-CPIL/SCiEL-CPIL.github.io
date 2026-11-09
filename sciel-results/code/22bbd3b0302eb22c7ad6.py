import time
import numpy as np
from bisect import bisect_left

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using NumPy vectorized sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline - 20:
        return [11, 6]

    # 2. Group numbers by sigma value using numpy argsort for speed
    n_vals = np.arange(1, MAX_N + 1, dtype=np.int64)
    s_vals = sigma[1:]  # sigma values for n=1..MAX_N
    
    # Get indices that sort by sigma value
    order = np.argsort(s_vals, kind='stable')
    sorted_s = s_vals[order]
    sorted_n = n_vals[order]
    
    # Find group boundaries where sigma value changes
    change_mask = np.diff(sorted_s) != 0
    boundaries = np.concatenate(([0], np.where(change_mask)[0] + 1, [MAX_N]))
    
    if time.time() > deadline - 15:
        return [11, 6]

    # 3. Search for best pair using exact rational arithmetic
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # Minimize this fraction; error must be > 0
    best_num = 1  # numerator of error fraction
    best_den = 3  # denominator: error = 1/3 for [11,6]
    best_pair = (11, 6)
    
    # Process each group
    for i in range(len(boundaries) - 1):
        if time.time() > deadline - 5:
            break
        start = boundaries[i]
        end = boundaries[i + 1]
        k = end - start
        if k < 2:
            continue
        
        group_n = sorted_n[start:end]  # sorted list of n values in this group
        
        # For each b in group, find a closest to 3b/2
        # Use integer arithmetic: minimize |2a - 3b| / (2b)
        for j in range(k):
            b = int(group_n[j])
            # a should be near 3b/2, so 2a target is 3b
            target_pos = bisect_left(group_n, 3 * b // 2)
            
            # Check candidates around target_pos
            candidates = set()
            if target_pos < k:
                candidates.add(target_pos)
            if target_pos > 0:
                candidates.add(target_pos - 1)
            if target_pos + 1 < k:
                candidates.add(target_pos + 1)
            if target_pos - 2 >= 0:
                candidates.add(target_pos - 2)
            if target_pos + 2 < k:
                candidates.add(target_pos + 2)
            
            for idx in candidates:
                a = int(group_n[idx])
                if a == b:
                    continue
                # Compute |2a - 3b| / (2b) exactly
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # invalid, error must be > 0
                den = 2 * b
                # Compare num/den with best_num/best_den
                # num/den < best_num/best_den iff num*best_den < best_num*den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]
