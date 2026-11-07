import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # Optimized Sieve
    # Process i in chunks based on the number of multiples
    # For i in (N/(k+1), N/k], the multiples are i, 2i, ..., k*i
    
    # We will iterate i from 2 upwards.
    # To avoid a 10M loop, we group i by their "density".
    
    # Simple optimization:
    # Loop only up to N/2.
    # For i > N/2, sigma[i] += i is the only update.
    # We can do this in one vectorized step.
    
    limit_half = N // 2
    if limit_half >= 2:
        for i in range(2, limit_half + 1):
            if time.time() > deadline - 10.0:
                return [23, 15]
            sigma[i::i] += i
            
    # Handle i > N/2
    if N > limit_half:
        sigma[limit_half + 1:] += np.arange(limit_half + 1, N + 1, dtype=np.int64)

    if time.time() > deadline - 5.0:
        return [23, 15]

    # 2. Search for best pair
    best_err = float('inf')
    best_pair = (23, 15)
    
    # Direct search
    # We process b in chunks to manage memory and allow deadline checks
    chunk_size = 1000000
    
    for b_start in range(1, N + 1, chunk_size):
        if time.time() > deadline - 5.0:
            break
            
        b_end = min(b_start + chunk_size, N + 1)
        b_arr = np.arange(b_start, b_end, dtype=np.int64)
        
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
                # Compute error: |2a - 3b| / (2b)
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
