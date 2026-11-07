import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin for 60s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    # This is the bottleneck. ~10-15s
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 15:
            # Fallback to a known good pair if sieve takes too long
            # [11, 6] is sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = 0.333
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 5:
        return [11, 6]

    # 2. Search for best pair by iterating error numerator k
    # Error = k / (2b)
    # We want to minimize this.
    # For a fixed k, we want the largest b such that sigma(a) == sigma(b)
    # where a = (3b + k) / 2.
    
    best_err = float('inf')
    best_a = 11
    best_b = 6
    max_b = 2 * N // 3  # a <= N => 1.5b <= N => b <= 2N/3
    
    # We iterate k from 1 upwards.
    # If k / (2 * max_b) > best_err, we can stop?
    # No, for larger k, we need larger b to compete.
    # But max_b is fixed.
    # So if k / (2 * max_b) > best_err, then even the best possible b for this k
    # will have an error larger than best_err.
    # Since k is increasing, k / (2 * max_b) will also increase.
    # So we can break the loop.
    
    k = 1
    while k <= 1000:  # Cap k to avoid infinite loop if no small error found
        # Minimum b required to beat current best_err
        # k / (2b) < best_err  =>  b > k / (2 * best_err)
        min_b = int(k / (2 * best_err)) + 1
        
        if min_b > max_b:
            break
            
        # Create array of b values
        # We want the largest b, so we can search from max_b down to min_b?
        # NumPy vectorization works on arrays.
        # Let's create the full range and filter.
        # To save memory/time, we can process in chunks if N is large, but N=10^7 is fine.
        
        b_vals = np.arange(min_b, max_b + 1, dtype=np.int64)
        
        # Check parity: 3b + k must be even
        # 3b + k is even => b and k have same parity (since 3 is odd)
        # If k is odd, b must be odd. If k is even, b must be even.
        # b_vals[0] is min_b.
        # We can step by 2.
        if (min_b % 2) != (k % 2):
            b_vals = b_vals[1::2]
        else:
            b_vals = b_vals[0::2]
            
        if len(b_vals) == 0:
            k += 1
            continue
            
        a_vals = (3 * b_vals + k) // 2
        
        # Check sigma equality
        # Ensure a_vals are within bounds (they should be, since b <= 2N/3 => a <= N)
        match_mask = sigma[a_vals] == sigma[b_vals]
        
        if np.any(match_mask):
            # Find the largest b (last index in b_vals)
            valid_indices = np.where(match_mask)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[-1]
                b = int(b_vals[best_idx])
                a = int(a_vals[best_idx])
                
                # Calculate error
                # err = abs(a/b - 1.5) = abs(2a - 3b) / (2b)
                # 2a - 3b = k (by construction, since a = (3b+k)/2 => 2a = 3b+k)
                err = k / (2 * b)
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
        k += 1
        
    return [int(best_a), int(best_b)]
