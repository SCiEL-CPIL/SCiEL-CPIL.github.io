import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N
    # Using a sieve method
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]
        
    # 2. Group numbers by their sigma value
    # sigma_to_numbers[s_val] = list of n such that sigma(n) = s_val
    # We only care about sigma values that appear more than once?
    # Actually, we can just build the full map.
    # To save memory/time, we can filter out sigma values with count 1,
    # but building the full map is simpler and fast enough.
    
    # Optimization: Instead of storing all numbers, we can just store the max number?
    # No, because for a specific a, we need to check if ANY b with same sigma exists.
    # But in our search loop, we iterate b and check a.
    # So we need: given a, does there exist b != a with sigma(b) = sigma(a)?
    # Or simpler: given b, does there exist a with sigma(a) = sigma(b)?
    # Since we iterate b, we can just check if sigma[b] is in the map and if the map contains any number other than b?
    # Actually, we are constructing a specific a from b.
    # So we just need to check: is sigma[a] == sigma[b]?
    # We don't need the map for the check itself, just the sigma array!
    # Wait, the map is not needed for the check `sigma[a] == sigma[b]`.
    # The map is only needed if we were searching for b given a, or finding pairs.
    # Here, we generate a from b and check the sigma array directly.
    # So we can skip building the dictionary entirely!
    
    # 3. Search for minimal error
    # Error = |2a - 3b| / (2b)
    # Let k = 2a - 3b.
    # We iterate k by magnitude.
    
    best_pair = (11, 6)
    best_error = abs(11/6 - 1.5) # 1/3
    
    # Max k to check. If we don't find anything with small k, we might need larger k.
    # But small k is much more likely to yield small error.
    # Let's try k up to 20.
    max_k = 20
    
    for k_mag in range(1, max_k + 1):
        for k in (-k_mag, k_mag):
            # Check time
            if time.time() > deadline:
                return [best_pair[0], best_pair[1]]
                
            # a = (3b + k) / 2
            # For a to be integer, 3b + k must be even.
            # 3b and b have same parity.
            # So b + k must be even.
            # If k is even, b must be even.
            # If k is odd, b must be odd.
            
            if k % 2 == 0:
                # b is even
                b_start = N if N % 2 == 0 else N - 1
                b_step = -2
            else:
                # b is odd
                b_start = N if N % 2 == 1 else N - 1
                b_step = -2
                
            # Iterate b downwards
            # We want to find the FIRST valid b (largest b) for this k.
            # Because error = |k| / (2b) decreases as b increases.
            for b in range(b_start, 0, b_step):
                # Calculate a
                num = 3 * b + k
                # Parity check is already handled by step, but let's be sure
                # num should be even
                if num % 2 != 0:
                    continue # Should not happen if step is correct
                a = num // 2
                
                # Check bounds for a
                if a < 1 or a > N:
                    continue
                    
                # Check sigma match
                if sigma[a] == sigma[b]:
                    # Valid pair found
                    # Calculate error
                    error = abs(k) / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_pair = (int(a), int(b))
                        # We found a valid pair for this k.
                        # Since we iterate k by magnitude, and b downwards,
                        # this is the best for this k.
                        # Can we break?
                        # If k_mag == 1, this is likely the global best or close.
                        # We can break the inner loop, but continue to next k?
                        # Actually, if we find a pair with k=1, error is ~ 1/(2b).
                        # Next k=2, error is ~ 2/(2b') = 1/b'.
                        # If b' is similar to b, k=1 is better.
                        # So we should continue searching for k=1 with smaller b? No, we want largest b.
                        # We found the largest b for this k.
                        # So we can break the inner loop for this k.
                        break
                        
        # Optional: If we found a pair with k=1, we can stop early?
        # Not necessarily, but it's a very good candidate.
        # Let's just continue to ensure we don't miss a k=1 with even larger b?
        # No, we iterated b downwards from N. The first hit is the max b.
        # So for each k, we get the best pair for that k.
        # We keep the global best.
        
    a, b = best_pair
    # Final validation
    if a == b or sigma[a] != sigma[b]:
        return [11, 6]
        
    return [a, b]
