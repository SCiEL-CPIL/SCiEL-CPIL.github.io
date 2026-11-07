import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 15:
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 5:
        return [11, 6]

    # 2. Search for best pair using k = 2a - 3b
    # Minimize |k| / (2b)
    # We check small |k| first.
    # For each k, we check all b.
    # To find the best, we need to check all k up to some limit.
    # If we find a pair with k=1 and b=N, error is 5e-8.
    # If we find a pair with k=2 and b=N, error is 1e-7.
    # So k=1 is better.
    # We should check k in order of |k|.
    
    best_num = float('inf') # This will store |k|
    best_den = 1 # This will store 2b
    best_a = 11
    best_b = 6
    
    # We want to minimize |k| / (2b)
    # Equivalent to minimizing |k| * (2b_best) vs |k_best| * (2b) ? No.
    # Just compare float values or use cross multiplication.
    # Let's store best_k_abs and best_b.
    # Error = best_k_abs / (2 * best_b)
    
    best_k_abs = abs(2 * 11 - 3 * 6) # 4
    best_b_val = 6
    
    # We will iterate k from 1 to K_max
    # And -k
    # We don't know K_max.
    # But if we find a solution with small k and large b, we can bound the rest.
    # Let's just iterate k up to 1000.
    # For each k, iterate b from N down to 1?
    # No, we need to check all b to ensure we don't miss a better one for larger k.
    # But if k is large, error is large.
    # If we have a candidate with error E, we only care about k < 2b E.
    
    # Strategy:
    # 1. Check k=1, -1 for all b.
    # 2. Check k=2, -2 for all b.
    # ...
    # Keep track of best_error.
    # Stop when k / (2 * 1) > best_error? i.e., k > 2 * best_error.
    # Since min b is 1, max error for a given k is k/2.
    # If k/2 > best_error, then no b can improve the score for this k or larger k.
    
    K_max = 10000
    
    for k in range(1, K_max + 1):
        # Check if k/2 > best_error
        # best_error = best_k_abs / (2 * best_b_val)
        # if k / 2.0 > best_k_abs / (2.0 * best_b_val):
        #    break
        if k * best_b_val > best_k_abs * 1: # k/best_b_val > best_k_abs -> k * best_b_val > best_k_abs
             # This condition is k > best_k_abs / best_b_val
             # Error for k with b=1 is k/2.
             # Current best error is best_k_abs / (2 * best_b_val).
             # If k/2 > best_k_abs / (2 * best_b_val), then k * best_b_val > best_k_abs.
             # In that case, even with b=1, the error is worse than current best.
             # And for any b > 1, error is smaller, but we are iterating k upwards.
             # Wait. For a fixed k, error decreases as b increases.
             # So for a fixed k, the best error is k / (2 * N).
             # If k / (2 * N) > best_error, then no b for this k can improve.
             # So break if k / (2 * N) > best_error.
             if k / (2.0 * N) > (best_k_abs / (2.0 * best_b_val)):
                 break
        
        # Check positive k
        # a = (3b + k) / 2
        # b must be such that 3b + k is even.
        # If k is even, b can be anything.
        # If k is odd, b must be odd.
        
        # We iterate b.
        # To be efficient, we can vectorize?
        # a = (3 * b + k) // 2
        # Check if (3*b + k) % 2 == 0
        # Check if a is in [1, N]
        # Check if sigma[a] == sigma[b]
        
        # Vectorized approach for a fixed k:
        # b_range = np.arange(1, N+1)
        # This creates a 10M array. Too slow for each k.
        
        # So we must loop in Python or use a smarter filter.
        # Loop in Python:
        # for b in range(1, N+1):
        #   ...
        # This is 10M iterations per k.
        # If we break early, it's fast.
        # If k=1, we do 10M iterations.
        # 10M iterations in Python take ~0.5-1s.
        # If we check k=1, 2, 3... up to 100, it's 100s. Too slow.
        
        # We need to optimize the inner loop.
        # Can we use NumPy for the inner loop?
        # Yes, but we need to avoid creating large arrays repeatedly.
        # We can pre-compute sigma.
        # For a fixed k, we want to find b such that:
        # 1. (3b + k) % 2 == 0
        # 2. 1 <= (3b + k) // 2 <= N
        # 3. sigma[(3b + k) // 2] == sigma[b]
        
        # Let's try to vectorize the check for a fixed k.
        # b_vals = np.arange(1, N+1)
        # This is 10M elements.
        # Condition 1:
        # If k is even: all b are valid parity-wise.
        # If k is odd: b must be odd.
        
        # Condition 2:
        # a = (3b + k) / 2
        # 1 <= a <= N  =>  2 <= 3b + k <= 2N
        # b >= (2 - k) / 3
        # b <= (2N - k) / 3
        
        # So we only need to check b in [b_min, b_max].
        
        b_min = max(1, int((2 - k) / 3) + 1)
        b_max = min(N, int((2 * N - k) / 3))
        
        if b_min > b_max:
            continue
            
        # Create array of b in range
        # To save memory/time, we can process in chunks?
        # Or just use the full range if it's not too big.
        # b_max - b_min is approx 2N/3 = 6.6M.
        
        # Let's try to vectorize.
        # But creating a 6M array for each k is expensive.
        
        # Alternative: Loop in Python but with local variables.
        # And break early if we find a good one?
        # No, we need the best for this k.
        # But if we find a pair with error E_k, and we already have E_best < E_k,
        # we don't need to find the *best* for this k, just *any* that is better than E_best?
        # No, we need the global best.
        # If we find one with error 1e-6, and current best is 1e-7, we ignore it.
        # If we find one with error 5e-8, we update.
        
        # So we can stop checking b for this k if the *minimum possible* error for this k
        # (which is k / (2 * b_max)) is greater than current best_error.
        # We already checked this at the start of the k loop.
        
        # So we must check all valid b for this k.
        
        # Let's use NumPy for the check.
        # b_arr = np.arange(b_min, b_max + 1)
        # This allocation is the cost.
        
        # Maybe we can just do the Python loop and hope it's fast enough for small k?
        # If k=1, we do 6M iterations. ~0.5s.
        # If we check k=1..10, ~5s.
        # This is feasible.
        
        sig = sigma
        for b in range(b_min, b_max + 1):
            num = 3 * b + k
            if num % 2 != 0:
                continue
            a = num // 2
            # a is guaranteed to be in [1, N] by b_min/b_max
            if sig[a] == sig[b]:
                # Found a pair
                # Error is k / (2b)
                # Compare with best
                # k / (2b) < best_k_abs / (2 * best_b_val)
                # k * best_b_val < best_k_abs * b
                if k * best_b_val < best_k_abs * b:
                    best_k_abs = k
                    best_b_val = b
                    best_a = a
                    best_b = b
                    # We found a better one.
                    # Can we break?
                    # For this k, error decreases as b increases.
                    # So if we iterate b from b_max down to b_min,
                    # the first match is the best for this k.
                    # Let's change iteration order to b_max down to b_min.
                    
        # Let's re-do the inner loop with b descending.
        
    # Re-implement inner loop with descending b
    # and early break if found.
    
    # Actually, the code above is getting complicated.
    # Let's stick to the simpler Grouping method which is proven to work,
    # but optimize the sieve if possible.
    # The grouping method found a score of 4M.
    # The k-method might find better, but is risky.
    # Let's try to combine:
    # 1. Fast Sieve (NumPy).
    # 2. Grouping and Search.
    # 3. If time permits, do a k-search for small k to polish.
    
    # But the prompt says "Rewrite the program from scratch with a stronger algorithm".
    # The k-search is stronger in theory.
    # Let's implement the k-search carefully.
    
    # Reset best
    best_k_abs = abs(2 * 11 - 3 * 6)
    best_b_val = 6
    best_a = 11
    best_b = 6
    
    # We will check k from 1 to K_limit
    # K_limit determined by current best.
    
    for k in range(1, 10000):
        # Check if k can improve
        # Min error for this k is k / (2 * N)
        # If k / (2 * N) >= best_k_abs / (2 * best_b_val), break
        if k * best_b_val >= best_k_abs * N:
            break
            
        # Determine b range
        b_min = max(1, (2 - k + 2) // 3) # ceil((2-k)/3)
        # (2-k) can be negative.
        # If k=1, (2-1)/3 = 1/3 -> 1.
        # If k=2, 0/3 = 0 -> 1.
        # If k=3, -1/3 -> 0 -> 1.
        # Formula: b >= (2-k)/3.
        # If 2-k > 0, b >= ceil((2-k)/3).
        # If 2-k <= 0, b >= 1.
        
        if 2 - k > 0:
            b_min = (2 - k + 2) // 3
            if b_min < 1: b_min = 1
        else:
            b_min = 1
            
        b_max = (2 * N - k) // 3
        if b_max > N: b_max = N
        
        if b_min > b_max:
            continue
            
        # Iterate b from b_max down to b_min
        # This ensures we find the largest b first, which gives the smallest error for this k.
        # If we find a match, it is the best for this k.
        # We update best and can break the inner loop.
        
        sig = sigma
        found = False
        for b in range(b_max, b_min - 1, -1):
            num = 3 * b + k
            if num % 2 != 0:
                continue
            a = num // 2
            if sig[a] == sig[b]:
                # Check if this is better than global best
                # k / (2b) < best_k_abs / (2 * best_b_val)
                if k * best_b_val < best_k_abs * b:
                    best_k_abs = k
                    best_b_val = b
                    best_a = a
                    best_b = b
                    found = True
                break # Best for this k found, break inner loop
                
    return [int(best_a), int(best_b)]
