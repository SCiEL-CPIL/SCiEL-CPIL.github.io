import numpy as np
import time

def construct():
    start_time = time.time()
    N = 10000000
    
    # 1. Compute sigma(n)
    sigma = np.ones(N + 1, dtype=np.int64)
    # Vectorized sieve
    # This loop is the bottleneck.
    # Optimization: only loop i up to N? Yes.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time
    if time.time() - start_time > 45:
        return [11, 6] # Fallback

    # 2. Sort by sigma
    # We only care about n >= 1
    vals_to_sort = np.arange(1, N + 1)
    sig_vals = sigma[1:N + 1]
    
    # Get indices that sort the sigma values
    sort_idx = np.argsort(sig_vals)
    sorted_sig = sig_vals[sort_idx]
    sorted_vals = vals_to_sort[sort_idx]
    
    # 3. Iterate through groups
    best_num = None
    best_den = None
    best_a = 11
    best_b = 6
    
    # Find boundaries of groups
    # np.where(sorted_sig[1:] != sorted_sig[:-1])[0] gives indices where sigma changes
    # We can use np.split or just iterate
    
    # To avoid creating many small arrays, we can find start/end indices
    # diff = np.diff(sorted_sig)
    # breaks = np.where(diff != 0)[0] + 1
    # starts = np.concatenate(([0], breaks))
    # ends = np.concatenate((breaks, [len(sorted_sig)]))
    
    # Let's just iterate with a pointer
    i = 0
    M = len(sorted_sig)
    while i < M:
        # Find end of group
        j = i
        current_sigma = sorted_sig[i]
        while j < M and sorted_sig[j] == current_sigma:
            j += 1
            
        group_size = j - i
        if group_size > 1:
            # Extract group values
            group_vals = sorted_vals[i:j]
            # group_vals is sorted by value? 
            # sorted_vals is the values n. 
            # sort_idx sorted by sigma. 
            # So group_vals contains n's with same sigma.
            # Are they sorted by n? No.
            # We need to sort them by n for binary search.
            group_vals.sort()
            
            # Iterate through b in group_vals
            # We can vectorize the target calculation, but searchsorted is per element.
            # Let's loop.
            for b in group_vals:
                # Target a = 1.5 b
                target = 1.5 * b
                # Find position in group_vals
                pos = np.searchsorted(group_vals, target)
                
                # Check pos and pos-1
                for idx in [pos, pos - 1]:
                    if 0 <= idx < group_size:
                        a = group_vals[idx]
                        if a == b:
                            continue
                        # Error |a/b - 1.5| = |2a - 3b| / (2b)
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            # Invalid score, but technically 0 error. 
                            # Task says "raw value of exactly 0 ... counts as invalid".
                            # So we should skip or handle? 
                            # "Minimize |a/b-3/2|". If 0 is achieved, it's invalid.
                            # We want the smallest NON-ZERO error.
                            # So skip num=0.
                            continue
                        den = 2 * b
                        
                        if best_num is None:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                        else:
                            # Compare num/den < best_num/best_den
                            # num * best_den < best_num * den
                            if num * best_den < best_num * den:
                                best_num = num
                                best_den = den
                                best_a = a
                                best_b = b
                                
        i = j
        
        # Time check
        if time.time() - start_time > 50:
            break
            
    return [int(best_a), int(best_b)]
