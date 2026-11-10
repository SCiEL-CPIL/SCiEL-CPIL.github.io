import numpy as np

def construct():
    max_n = 1000000
    window_size = 24
    limit = max_n + window_size + 10
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # 1. Compute g[m] = product of prime factors < 24
    # g[m] = prod_{p < 24} p^{v_p(m)}
    # We use the sieve method: for each prime p, and each power p^k, 
    # multiply g[j] by p for all j multiple of p^k.
    
    g = [1] * (limit + 1)
    
    for p in primes:
        power = p
        while power <= limit:
            for j in range(power, limit + 1, power):
                g[j] *= p
            power *= p
            
    # 2. Sliding Window with Array-based Counting
    # Max value of g[m] is m (if m is 23-smooth).
    # Max m is ~1e6. So array size 1e6+1 is fine.
    max_g_val = limit # Safe upper bound
    
    counts = [0] * (max_g_val + 1)
    
    # Initialize window for n=1000: indices 1001 to 1024
    start_idx = 1001
    end_idx = 1001 + window_size # exclusive, so 1025
    
    current_distinct = 0
    
    for i in range(start_idx, end_idx):
        val = g[i]
        if counts[val] == 0:
            current_distinct += 1
        counts[val] += 1
            
    min_distinct = current_distinct
    best_n = 1000
    
    # Slide
    # n goes from 1000 to max_n - 1
    # Window for n: [n+1, n+24]
    # Window for n+1: [n+2, n+25]
    # Remove g[n+1], Add g[n+25]
    
    for n in range(1000, max_n):
        # Remove index n+1
        out_val = g[n+1]
        counts[out_val] -= 1
        if counts[out_val] == 0:
            current_distinct -= 1
            
        # Add index n+25
        in_val = g[n+25]
        if counts[in_val] == 0:
            current_distinct += 1
        counts[in_val] += 1
            
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n + 1
            
    return best_n
