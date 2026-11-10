import numpy as np
import time

def construct():
    start_time = time.time()
    max_n = 10_000_000
    deadline = start_time + 50.0  # Leave 10s margin
    
    # 1. Compute sigma(n) for n=1..max_n
    sigma = np.ones(max_n + 1, dtype=np.int64)
    # Sieve for sigma
    # sigma(n) = sum_{d|n} d
    # We can iterate d from 1 to max_n, and add d to all multiples of d.
    # This is O(N log N).
    for i in range(1, max_n + 1):
        sigma[i::i] += i
        
    # 2. Search for best pair
    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # We iterate k = |2a - 3b|
    # Error = k / (2b)
    # For a given k, max b is approx 10^7.
    # Min error for k is approx k / (2 * 10^7)
    # If k / (2 * 10^7) >= best_error, we can stop.
    
    # Parity check:
    # 2a = 3b + k => 3b + k even => b and k opposite parity.
    # 2a = 3b - k => 3b - k even => b and k opposite parity.
    
    for k in range(1, 2000): # Limit k to avoid infinite loop, though break condition handles it
        # Check if further k can improve
        min_possible_error = k / (2.0 * max_n)
        if min_possible_error >= best_error:
            break
            
        if time.time() > deadline:
            break
            
        # Case 1: 2a = 3b + k => a = (3b + k) // 2
        # b must have parity opposite to k
        # b_max such that a <= max_n: (3b + k) <= 2*max_n => b <= (2*max_n - k) // 3
        b_limit = (2 * max_n - k) // 3
        if b_limit > max_n:
            b_limit = max_n
            
        # Start b from b_limit, adjust for parity
        # b % 2 != k % 2
        start_b = b_limit
        if start_b % 2 == k % 2:
            start_b -= 1
            
        for b in range(start_b, 0, -2):
            a = (3 * b + k) // 2
            if a > max_n:
                continue
            if a < 1:
                break # As b decreases, a decreases. If a < 1, no more valid a.
            
            if sigma[a] == sigma[b]:
                error = k / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                # Since b is decreasing, error is increasing for this k.
                # We can break if error >= best_error? No, best_error might be updated.
                # But for this k, subsequent b will have larger error.
                # So we only care if this specific (a,b) is better than current best.
                # If it is, we update. If not, we don't.
                # Can we break early?
                # If k/(2*b) >= best_error, then for smaller b, error is even larger.
                # So yes, break.
                if error >= best_error:
                    break
                    
        # Case 2: 2a = 3b - k => a = (3b - k) // 2
        # b must have parity opposite to k
        # a >= 1 => 3b - k >= 2 => b >= (k + 2) // 3
        # a <= max_n => 3b - k <= 2*max_n => b <= (2*max_n + k) // 3
        b_limit = (2 * max_n + k) // 3
        if b_limit > max_n:
            b_limit = max_n
            
        start_b = b_limit
        if start_b % 2 == k % 2:
            start_b -= 1
            
        for b in range(start_b, 0, -2):
            a = (3 * b - k) // 2
            if a > max_n:
                continue
            if a < 1:
                break
                
            if sigma[a] == sigma[b]:
                error = k / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                if error >= best_error:
                    break

    return [int(best_a), int(best_b)]
