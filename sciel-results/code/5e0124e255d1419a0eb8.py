import numpy as np

def construct():
    N = 10_000_000
    # Compute sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    best_err = float('inf')
    best_a, best_b = 1, 1
    
    # Search offsets
    # We check a = round(1.5 * b) + k
    # k range: -200 to 200
    # To save time, we can split into chunks of b?
    # No, vectorized on full N is fine if memory allows.
    
    # Pre-allocate b_vals
    b_vals = np.arange(1, N + 1, dtype=np.int64)
    
    # We iterate k
    for k in range(-200, 201):
        # Calculate a_vals
        # 1.5 * b_vals might be float, round to int
        # Using integer arithmetic: a = round(3*b/2) + k
        # 3*b/2 is exact if b even. If b odd, 3*b/2 is .5
        # np.round(1.5 * b_vals) is fine.
        
        a_vals = np.round(1.5 * b_vals).astype(np.int64) + k
        
        # Mask valid a
        mask = (a_vals >= 1) & (a_vals <= N) & (a_vals != b_vals)
        
        if not np.any(mask):
            continue
            
        a_valid = a_vals[mask]
        b_valid = b_vals[mask]
        
        # Check sigma equality
        # This creates new arrays, memory heavy?
        # sigma[a_valid] and sigma[b_valid]
        if len(a_valid) == 0:
            continue
            
        sigma_a = sigma[a_valid]
        sigma_b = sigma[b_valid]
        
        match_mask = (sigma_a == sigma_b)
        
        if not np.any(match_mask):
            continue
            
        a_match = a_valid[match_mask]
        b_match = b_valid[match_mask]
        
        # Calculate error
        # error = |a/b - 1.5|
        # To avoid float division issues or overflow, use rational?
        # a/b - 1.5 = (2a - 3b) / 2b
        # abs(2a - 3b) / (2b)
        
        num = np.abs(2 * a_match - 3 * b_match)
        den = 2 * b_match
        
        # We want min num/den
        # num can be 0? If 2a = 3b, then a/b = 1.5.
        # If num == 0, error is 0. Invalid.
        # So filter num > 0
        
        non_zero_mask = (num > 0)
        if not np.any(non_zero_mask):
            continue
            
        num_valid = num[non_zero_mask]
        den_valid = den[non_zero_mask]
        a_valid_match = a_match[non_zero_mask]
        b_valid_match = b_match[non_zero_mask]
        
        # Find min ratio
        # ratio = num_valid / den_valid
        # We can use np.argmin(num_valid / den_valid)
        # But division creates float array.
        
        # To be safe and fast, we can just compute the float error
        errors = num_valid.astype(np.float64) / den_valid.astype(np.float64)
        
        min_idx = np.argmin(errors)
        min_err = errors[min_idx]
        
        if min_err < best_err:
            best_err = min_err
            best_a = int(a_valid_match[min_idx])
            best_b = int(b_valid_match[min_idx])
            
    return [best_a, best_b]
