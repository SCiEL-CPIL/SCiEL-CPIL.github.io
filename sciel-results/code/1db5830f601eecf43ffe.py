import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 15:
        return [11, 6]

    best_err = float('inf')
    best_pair = (11, 6) # Fallback
    
    # We search for pairs where 2a - 3b = m or 3b - 2a = m
    # |2a - 3b| = m
    # We want to minimize m / (2b)
    
    # Iterate m from 1 upwards
    # For each m, find the largest b such that sigma(a) == sigma(b)
    # a = (3b + m) / 2  (Case 1: 2a - 3b = m)
    # a = (3b - m) / 2  (Case 2: 3b - 2a = m)
    
    # We can combine these by checking both signs for a given m.
    # Actually, it's easier to just iterate m and check both cases.
    
    max_m = 200  # Search up to m=200
    
    for m in range(1, max_m + 1):
        if time.time() > deadline - 5:
            break
            
        # Case 1: 2a - 3b = m => 2a = 3b + m => a = (3b + m) / 2
        # 3b + m must be even.
        # If m is odd, b must be odd.
        # If m is even, b must be even.
        
        if m % 2 == 1:
            b_start = 1
            b_step = 2
        else:
            b_start = 2
            b_step = 2
            
        # Max b such that a <= N
        # (3b + m) / 2 <= N => 3b + m <= 2N => b <= (2N - m) / 3
        b_max = min(N, (2 * N - m) // 3)
        
        if b_max >= b_start:
            # Generate b values
            # We want to find the LARGEST b.
            # So we can generate b in descending order?
            # Or generate ascending and take the last valid one.
            # Generating ascending is easier with numpy.
            
            # Number of elements
            n_elems = (b_max - b_start) // b_step + 1
            if n_elems > 0:
                b_vals = np.arange(b_start, b_max + 1, b_step, dtype=np.int64)
                a_vals = (3 * b_vals + m) // 2
                
                # Ensure a is valid (1 <= a <= N)
                # a_vals are calculated such that a <= N by b_max.
                # a >= 1 is guaranteed for b >= 1, m >= 1.
                
                # Check sigma equality
                # Need to handle a_vals as indices
                # Note: a_vals might exceed N if b_max calculation was slightly off, but we used floor.
                # Also a_vals must be <= N.
                
                valid_mask = (a_vals <= N) & (a_vals >= 1)
                
                if np.any(valid_mask):
                    # We only check valid ones
                    # But we want the largest b, which corresponds to the largest index in b_vals
                    # So we can just check the condition and find the last True.
                    
                    sig_a = sigma[a_vals[valid_mask]]
                    sig_b = sigma[b_vals[valid_mask]]
                    
                    match_mask = sig_a == sig_b
                    
                    if np.any(match_mask):
                        # Find the index of the last match
                        # Since b_vals is sorted ascending, the last match has the largest b
                        # We need to map back to the original indices or just find the value
                        # Easier: find the index in the valid_mask array
                        match_indices = np.where(match_mask)[0]
                        if len(match_indices) > 0:
                            last_idx = match_indices[-1]
                            # Map last_idx back to the original b_vals index
                            # valid_mask is a boolean array.
                            # We need the index in b_vals corresponding to the last True in valid_mask & match_mask?
                            # No, match_mask is computed on the subset.
                            # Let's just compute the full mask on b_vals directly to avoid mapping complexity.
                            
                            # Re-evaluate:
                            # mask_full = (sigma[a_vals] == sigma[b_vals]) & (a_vals <= N) & (a_vals >= 1) & (a_vals != b_vals)
                            # If any, take b_vals[np.where(mask_full)[0][-1]]
                            
                            mask_full = (sigma[a_vals] == sigma[b_vals]) & (a_vals <= N) & (a_vals >= 1) & (a_vals != b_vals)
                            
                            if np.any(mask_full):
                                indices = np.where(mask_full)[0]
                                best_idx = indices[-1]
                                b_curr = int(b_vals[best_idx])
                                a_curr = int(a_vals[best_idx])
                                
                                err = m / (2.0 * b_curr)
                                if err < best_err:
                                    best_err = err
                                    best_pair = (a_curr, b_curr)
        
        # Case 2: 3b - 2a = m => 2a = 3b - m => a = (3b - m) / 2
        # 3b - m must be even.
        # If m is odd, b must be odd.
        # If m is even, b must be even.
        
        if m % 2 == 1:
            b_start = 1
            b_step = 2
        else:
            b_start = 2
            b_step = 2
            
        # Max b such that a >= 1
        # (3b - m) / 2 >= 1 => 3b - m >= 2 => b >= (m + 2) / 3
        b_min = max(b_start, (m + 2 + b_step - 1) // b_step) # Ensure b_min has correct parity
        
        # Wait, b_min calculation:
        # We need b such that 3b - m >= 2.
        # b >= (m+2)/3.
        # And b has parity of m.
        
        b_min_val = (m + 2) // 3
        if b_min_val % 2 != m % 2:
            b_min_val += 1
        b_min_val = max(b_min_val, b_start)
        
        b_max = N # No upper bound from a <= N, as a = (3b-m)/2 < 1.5b < 1.5N, but a must be <= N.
        # a <= N => (3b - m) / 2 <= N => 3b - m <= 2N => b <= (2N + m) / 3.
        b_max = min(N, (2 * N + m) // 3)
        
        if b_max >= b_min_val:
            n_elems = (b_max - b_min_val) // b_step + 1
            if n_elems > 0:
                b_vals = np.arange(b_min_val, b_max + 1, b_step, dtype=np.int64)
                a_vals = (3 * b_vals - m) // 2
                
                mask_full = (sigma[a_vals] == sigma[b_vals]) & (a_vals <= N) & (a_vals >= 1) & (a_vals != b_vals)
                
                if np.any(mask_full):
                    indices = np.where(mask_full)[0]
                    best_idx = indices[-1]
                    b_curr = int(b_vals[best_idx])
                    a_curr = int(a_vals[best_idx])
                    
                    err = m / (2.0 * b_curr)
                    if err < best_err:
                        best_err = err
                        best_pair = (a_curr, b_curr)

    if best_pair is not None and best_err < float('inf'):
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
