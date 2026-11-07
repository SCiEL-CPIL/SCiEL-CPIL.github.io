import numpy as np
import time

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    N = 10000000

    # 1. Compute sigma(n) for all n up to N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # Optimization: split loop? No, NumPy slice assignment is fast enough.
    # for i in range(1, N + 1): sigma[i::i] += i
    # This loop runs N times. Each slice operation is O(N/i).
    # Total operations ~ N * sum(1/i) ~ N log N.
    # For N=10^7, this is manageable in NumPy if not too many Python loop iterations.
    # However, 10^7 Python loop iterations is slow (~5-10s).
    # We can optimize by only iterating i up to N, but the slice update is vectorized.
    
    # To speed up the sieve loop itself, we can't easily vectorize the loop index i.
    # But we can break early if time is tight.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            # Fallback if sieve is too slow
            return [11, 6]
        sigma[i::i] += i

    if time.time() > deadline - 10.0:
        return [11, 6]

    # 2. Vectorized Search
    # We want to find b in [1, N] such that there exists a in [1, N] with sigma(a) == sigma(b)
    # and a is close to 1.5 * b.
    
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    sb_arr = sigma[b_arr]
    
    # Candidate a values
    a1 = (3 * b_arr) // 2
    a2 = a1 + 1
    
    # Validity masks
    # a must be in [1, N] and a != b
    valid_a1 = (a1 >= 1) & (a1 <= N) & (a1 != b_arr)
    valid_a2 = (a2 >= 1) & (a2 <= N) & (a2 != b_arr)
    
    # Check sigma match
    # We need to be careful with indexing.
    # sigma is 1-indexed.
    
    # For a1
    # We can't directly index sigma with an array that has invalid values without masking or padding.
    # Let's create safe indices.
    a1_safe = np.where(valid_a1, a1, 1) # Use 1 as dummy, sigma[1]=1
    a2_safe = np.where(valid_a2, a2, 1)
    
    match_a1 = valid_a1 & (sigma[a1_safe] == sb_arr)
    match_a2 = valid_a2 & (sigma[a2_safe] == sb_arr)
    
    # Calculate errors for matches
    # Error = |2a - 3b| / (2b)
    
    # For a1
    err_a1 = np.full(N, np.inf, dtype=np.float64)
    if np.any(match_a1):
        # 2*a1 - 3*b
        # Note: a1 = floor(1.5b). 
        # If b is even, a1 = 1.5b exactly. 2a1 - 3b = 0.
        # If b is odd, a1 = (3b-1)/2. 2a1 - 3b = -1.
        # So error is either 0 or 1/(2b).
        # We must exclude 0.
        
        num_a1 = np.abs(2 * a1_safe[match_a1] - 3 * b_arr[match_a1])
        # Exclude 0
        nonzero_mask = num_a1 > 0
        if np.any(nonzero_mask):
            valid_indices_a1 = np.where(match_a1)[0]
            valid_indices_a1 = valid_indices_a1[nonzero_mask]
            # Calculate error for these
            b_vals = b_arr[valid_indices_a1]
            a_vals = a1_safe[valid_indices_a1]
            errs = np.abs(2 * a_vals - 3 * b_vals) / (2.0 * b_vals)
            # Update global min
            min_err = np.min(errs)
            min_idx = np.argmin(errs)
            best_a = int(a_vals[min_idx])
            best_b = int(b_vals[min_idx])
            
            # We should also check a2.
            # Let's combine both.
            
    # Let's do a cleaner combined approach to avoid logic errors.
    
    # Create arrays for all candidates
    # We will compute errors for all valid a1 and a2 matches and take the global min.
    
    # Reset
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Check a1 matches
    if np.any(match_a1):
        idxs = np.where(match_a1)[0]
        # Filter out num == 0
        num_vals = np.abs(2 * a1_safe[idxs] - 3 * b_arr[idxs])
        valid_idxs = idxs[num_vals > 0]
        if len(valid_idxs) > 0:
            b_vals = b_arr[valid_idxs]
            a_vals = a1_safe[valid_idxs]
            errs = (num_vals[num_vals > 0]) / (2.0 * b_vals)
            min_idx = np.argmin(errs)
            if errs[min_idx] < best_error:
                best_error = errs[min_idx]
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])
                
    # Check a2 matches
    if np.any(match_a2):
        idxs = np.where(match_a2)[0]
        num_vals = np.abs(2 * a2_safe[idxs] - 3 * b_arr[idxs])
        valid_idxs = idxs[num_vals > 0]
        if len(valid_idxs) > 0:
            b_vals = b_arr[valid_idxs]
            a_vals = a2_safe[valid_idxs]
            errs = (num_vals[num_vals > 0]) / (2.0 * b_vals)
            min_idx = np.argmin(errs)
            if errs[min_idx] < best_error:
                best_error = errs[min_idx]
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])
                
    # Fallback if no better pair found
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
