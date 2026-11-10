import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N using a sieve
    # Using the standard O(N log N) sieve is fast enough in Python with NumPy?
    # Actually, pure Python loop is slow. NumPy is faster.
    # sigma = np.zeros(N + 1, dtype=np.int64)
    # for i in range(1, N + 1):
    #     sigma[i::i] += i
    # This is ~ N log N operations. 10^7 * 16 ~ 1.6 * 10^8 operations.
    # In NumPy, this is vectorized and should take a few seconds.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 1_000_000 == 0:
            if time.time() > deadline - 20:
                # Fallback to a known pair if sieve takes too long
                # 11, 6 is from seed: sigma(11)=12, sigma(6)=12.
                # |11/6 - 1.5| = |1.833 - 1.5| = 0.333
                return [11, 6]
                
    if time.time() > deadline - 15:
        return [11, 6]
        
    # 2. Direct Arithmetic Search
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # ~0.333
    
    # We iterate k = |2a - 3b|
    # We want small k and large b.
    # Error = k / (2b)
    
    # Max k to check?
    # If best_err is updated, we can prune.
    # Initially best_err is ~0.33.
    # k/(2N) < 0.33 => k < 0.66 * 2 * 10^7 = 1.32 * 10^7.
    # That's too many k's if we iterate b for each.
    
    # However, we expect to find a solution with small k and large b quickly.
    # If we find k=1, b~N, err ~ 5e-8.
    # Then k/(2N) < 5e-8 => k < 1.
    # So we stop after k=1.
    
    # So the risk is only if small k has NO solution with large b.
    # We can limit the inner loop iterations or add a timeout per k.
    
    max_k = 1000 # Arbitrary upper bound for k to start with?
    # Actually, let's just run until time limit or pruning condition met.
    
    for k in range(1, 10000):
        # Pruning: if minimum possible error for this k is worse than best, break
        if k / (2.0 * N) > best_err:
            break
            
        # Check time
        if time.time() > deadline - 5:
            break
            
        # Iterate b from N down to 1
        # We want the largest b such that:
        # 1. (3b + k) is even
        # 2. a = (3b + k) / 2 <= N
        # 3. a >= 1
        # 4. sigma[a] == sigma[b]
        
        # a = (3b+k)/2 <= N => 3b+k <= 2N => b <= (2N-k)/3
        # So max b is min(N, (2N-k)//3)
        
        max_b = min(N, (2*N - k) // 3)
        
        # We need 3b + k to be even.
        # If k is even, 3b must be even => b even.
        # If k is odd, 3b must be odd => b odd.
        
        # Start b at max_b. Adjust to correct parity.
        b = max_b
        if (3 * b + k) % 2 != 0:
            b -= 1
            
        # Now iterate b downwards with step 2 to maintain parity
        # We stop when a < 1 => (3b+k)/2 < 1 => 3b+k < 2 => b < (2-k)/3
        # For k >= 1, b >= 0 is enough, but b must be positive.
        # If k=1, 3b+1 >= 2 => 3b >= 1 => b >= 1.
        # If k=2, 3b+2 >= 2 => b >= 0. But b>=1.
        
        # We also need to check time inside the loop if it's long
        # But if we find a solution, we break.
        # If we don't find a solution, we scan up to N/2 values.
        # This might be too slow for many k's.
        
        # Optimization: We can sample or limit the search depth if time is tight?
        # Or just rely on the fact that solutions are likely to exist for small k.
        
        for b_val in range(b, 0, -2):
            a_val = (3 * b_val + k) // 2
            
            # Check bounds
            if a_val > N or a_val < 1:
                continue
                
            # Check sigma equality
            if sigma[a_val] == sigma[b_val]:
                # Found a valid pair for this k
                # Since we iterate b downwards, this is the largest b for this k
                # So this is the best error for this k
                err = k / (2.0 * b_val)
                if err < best_err:
                    best_err = err
                    best_a = a_val
                    best_b = b_val
                break # Break inner loop, next k
        
        # If we didn't break inner loop, no solution for this k
        # Continue to next k
        
    return [int(best_a), int(best_b)]
