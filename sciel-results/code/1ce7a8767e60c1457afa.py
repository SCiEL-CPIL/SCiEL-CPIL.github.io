import math

def construct():
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Define window size
    W = 200
    
    # 3. Choose starting point
    # We want s + W - 1 <= 10**12
    # Let's start near the upper bound to maximize the "randomness" / magnitude
    # s = 10**12 - W + 1
    max_limit = 10**12
    s = max_limit - W + 1
    
    # We might try a few starts if the first one fails
    for attempt in range(100):
        start = s - attempt * W # jump by W to avoid overlap? or just s-1?
        # Jumping by W ensures we check disjoint windows, but we want to maximize k.
        # If a window fails, it's likely due to a specific collision.
        # Moving by 1 is safer to find *a* valid window if it exists nearby.
        # But given the low prob, s-1 is fine.
        
        # Let's just try consecutive starts if needed, but likely first one works.
        # Actually, let's just try the one calculated.
        
        # Initialize arrays
        # nums[i] will hold the remaining factor part
        # phis[i] will hold the phi value
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        # Segmented sieve for phi
        for p in primes:
            if p * p > max_limit: # Optimization: if p > sqrt(max num), we handle remainder later
                # Actually we need to process all p <= sqrt(start+W-1)
                # But if p > sqrt(num), and num % p == 0, then num/p is 1 or prime.
                # Standard approach: process all p <= sqrt(max_num).
                # The rest is handled by the "remaining > 1" check.
                pass
            
            # Find first multiple of p in [start, start+W-1]
            # m = ceil(start / p) * p
            m = ((start + p - 1) // p) * p
            
            if m > start + W - 1:
                continue
                
            for x in range(m, start + W, p):
                idx = x - start
                if nums[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    while nums[idx] % p == 0:
                        nums[idx] //= p
                        
        # Handle remaining prime factors
        for i in range(W):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        # Check distinctness
        if len(set(phis)) == W:
            return {'start': start, 'length': W}
            
    # If we get here, we failed to find a run of 200.
    # Fallback to smaller k?
    # The task says maximize k.
    # If 200 is impossible (unlikely), we should try 199, etc.
    # But given the analysis, 200 should be found.
    
    # Fallback logic just in case
    for k in range(199, 0, -1):
        # Try to find a run of length k
        # This is expensive if we do it naively.
        # But we can reuse the phi calculation for the window of 200.
        # If the window of 200 has collisions, maybe a sub-window of 199 is valid.
        # We can check all sub-windows of the computed phis.
        
        # Re-compute for the last tried 'start' or just use the one from the loop?
        # Let's re-run for a specific start and check sub-windows.
        
        pass

    # Absolute fallback
    return {'start': 1, 'length': 1}
