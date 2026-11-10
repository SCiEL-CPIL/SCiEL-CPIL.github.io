import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using vectorized operations
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # Get all multiples of p
        indices = np.arange(p, max_val + 1, p)
        if len(indices) == 0:
            continue
        
        # We need to multiply small_factor[indices] by p^k where k is the exponent of p in the index
        # Instead of dividing the indices (which requires storing a copy), 
        # we can just count the power of p in each index.
        
        # Create a copy of indices to modify
        vals = indices.copy()
        exponents = np.zeros(len(vals), dtype=np.int64)
        
        # Count exponents
        # Loop while any val is divisible by p
        # This loop runs at most log_p(max_val) times, which is small (e.g., 20 for p=2)
        while True:
            mask = vals % p == 0
            if not np.any(mask):
                break
            exponents[mask] += 1
            vals[mask] //= p
            
        # Calculate p^exponents
        # np.power is vectorized
        multipliers = np.power(p, exponents)
        
        # Update small_factor
        small_factor[indices] *= multipliers
        
    # Scan for the best n
    best_score = 25
    best_n = 1000
    
    # We scan n from 1000 to 1000000
    # Window is n+1 to n+24
    # small_factor[n+1 : n+25]
    
    # To speed up the scan, we can check time periodically
    # But 1M iterations with np.unique should be fast enough.
    
    for n in range(1000, max_n + 1):
        if n % 50000 == 0 and time.time() > deadline:
            break
            
        window = small_factor[n + 1 : n + 25]
        # Count distinct values
        # np.unique sorts, which is O(k log k) for k=24. 
        # Alternative: len(set(window)) might be faster for small k in some Python versions, 
        # but np.unique is generally robust.
        # Let's use np.unique.
        count = len(np.unique(window))
        
        if count < best_score:
            best_score = count
            best_n = n
            if best_score == 1:
                break  # Can't do better than 1
                
    return best_n
