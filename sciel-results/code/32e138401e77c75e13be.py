import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This is the bottleneck.
    # We can try to speed it up.
    # Using np.add.at? No, slicing is faster.
    
    # Optimization: 
    # The loop `for i in range(1, N+1): sigma[i::i] += i` 
    # is O(N log N).
    # For large i, the slice is small.
    # We can split the loop?
    # No, NumPy handles it well.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 10.0:
            # Fallback
            return [23, 15]
        sigma[i::i] += i
        
    if time.time() > deadline - 5.0:
        return [23, 15]

    # 2. Search for best pair
    best_err = float('inf')
    best_pair = (23, 15)
    
    # Direct search
    # Iterate b
    # We can vectorize parts of this?
    # b = np.arange(1, N+1)
    # a_target = 1.5 * b
    # a1 = np.floor(a_target).astype(int)
    # a2 = a1 + 1
    # Check bounds
    # valid1 = (a1 >= 1) & (a1 <= N) & (a1 != b)
    # valid2 = (a2 >= 1) & (a2 <= N) & (a2 != b)
    # sigma_a1 = sigma[a1[valid1]]
    # sigma_b1 = sigma[b[valid1]]
    # matches1 = (sigma_a1 == sigma_b1)
    # ...
    # This vectorization might be faster.
    
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    a_target = 1.5 * b_arr
    a1 = np.floor(a_target).astype(np.int64)
    a2 = a1 + 1
    
    # Mask valid a1
    mask1 = (a1 >= 1) & (a1 <= N) & (a1 != b_arr)
    # Mask valid a2
    mask2 = (a2 >= 1) & (a2 <= N) & (a2 != b_arr)
    
    # Process a1
    if np.any(mask1):
        b_valid = b_arr[mask1]
        a_valid = a1[mask1]
        sb = sigma[b_valid]
        sa = sigma[a_valid]
        matches = (sa == sb)
        if np.any(matches):
            b_match = b_valid[matches]
            a_match = a_valid[matches]
            # Compute error
            # err = |2a - 3b| / (2b)
            num = np.abs(2 * a_match - 3 * b_match)
            den = 2.0 * b_match
            errs = num / den
            min_idx = np.argmin(errs)
            min_err = errs[min_idx]
            min_a = int(a_match[min_idx])
            min_b = int(b_match[min_idx])
            
            if min_err < best_err:
                best_err = min_err
                best_pair = (min_a, min_b)
                
    # Process a2
    if np.any(mask2):
        b_valid = b_arr[mask2]
        a_valid = a2[mask2]
        sb = sigma[b_valid]
        sa = sigma[a_valid]
        matches = (sa == sb)
        if np.any(matches):
            b_match = b_valid[matches]
            a_match = a_valid[matches]
            num = np.abs(2 * a_match - 3 * b_match)
            den = 2.0 * b_match
            errs = num / den
            min_idx = np.argmin(errs)
            min_err = errs[min_idx]
            min_a = int(a_match[min_idx])
            min_b = int(b_match[min_idx])
            
            if min_err < best_err:
                best_err = min_err
                best_pair = (min_a, min_b)
                
    return [int(best_pair[0]), int(best_pair[1])]
