import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Define values for first and second half
    # i=1..15 and i=16..30
    vals_A = [L // i for i in range(1, 16)]
    vals_B = [L // i for i in range(16, 31)]
    
    # 3. Generate sums and indices for half A
    def generate_sums_indices(vals):
        sums = [0]
        indices = [0]
        for v in vals:
            new_sums = []
            new_indices = []
            # Preallocate lists for speed?
            # List append is fast enough for 14M
            for s, k in zip(sums, indices):
                new_sums.append(s)
                new_indices.append(k)
                new_sums.append(s + v)
                new_indices.append(k * 3 + 1)
                new_sums.append(s - v)
                new_indices.append(k * 3 + 2)
            sums = new_sums
            indices = new_indices
            if time.time() > deadline:
                break
        return sums, indices

    sums_A, indices_A = generate_sums_indices(vals_A)
    if time.time() > deadline:
        # Fallback to seed
        return [0] * 28 + [1, -1]

    # Build dict: sum -> index (first occurrence)
    dict_sum_to_idx_A = {}
    for s, k in zip(sums_A, indices_A):
        if s not in dict_sum_to_idx_A:
            dict_sum_to_idx_A[s] = k
            
    # Sort sums_A for binary search
    sums_A_sorted = sorted(sums_A)
    np_sums_A = np.array(sums_A_sorted, dtype=np.int64)
    
    # 4. Generate sums and indices for half B
    sums_B, indices_B = generate_sums_indices(vals_B)
    if time.time() > deadline:
        # Fallback to seed
        return [0] * 28 + [1, -1]
        
    np_sums_B = np.array(sums_B, dtype=np.int64)
    
    # 5. Vectorized search for closest pair
    # We want min |a + b| != 0
    # a in np_sums_A, b in np_sums_B
    # For each b, find a closest to -b
    
    queries = -np_sums_B
    idxs = np.searchsorted(np_sums_A, queries)
    
    # Check idxs and idxs - 1
    # Handle bounds
    idxs_clipped = np.clip(idxs, 1, len(np_sums_A) - 1)
    
    diffs1 = np_sums_A[idxs_clipped] + np_sums_B
    diffs2 = np_sums_A[idxs_clipped - 1] + np_sums_B
    
    # Combine diffs
    all_diffs = np.concatenate([diffs1, diffs2])
    
    # Filter nonzero
    nonzero_mask = all_diffs != 0
    if not np.any(nonzero_mask):
        return [0] * 28 + [1, -1]
        
    nonzero_diffs = all_diffs[nonzero_mask]
    abs_diffs = np.abs(nonzero_diffs)
    min_idx = np.argmin(abs_diffs)
    best_diff = nonzero_diffs[min_idx]
    best_abs = abs_diffs[min_idx]
    
    # Determine which diff (1 or 2) and which b index
    # If min_idx < len(diffs1), it came from diffs1
    # Else from diffs2
    if min_idx < len(diffs1):
        b_idx = min_idx
        a_idx = idxs_clipped[b_idx]
        source = 1
    else:
        b_idx = min_idx - len(diffs1)
        a_idx = idxs_clipped[b_idx] - 1
        source = 2
        
    best_sum_A = np_sums_A[a_idx]
    best_sum_B = np_sums_B[b_idx]
    
    # 6. Recover indices
    idx_A = dict_sum_to_idx_A[best_sum_A]
    idx_B = indices_B[b_idx]
    
    # 7. Decode indices to coefficients
    def decode_index(k, n):
        coeffs = [0] * n
        for i in range(n):
            d = k % 3
            k //= 3
            if d == 1:
                coeffs[i] = 1
            elif d == 2:
                coeffs[i] = -1
            else:
                coeffs[i] = 0
        return coeffs
    
    coeffs_A = decode_index(idx_A, 15)
    coeffs_B = decode_index(idx_B, 15)
    
    result = coeffs_A + coeffs_B
    
    # Verify
    # sum(c[i]/(i+1) for i in range(30))
    # Should be best_diff / L
    
    return result
