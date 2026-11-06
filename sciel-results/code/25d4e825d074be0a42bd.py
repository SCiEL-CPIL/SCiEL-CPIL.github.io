import time
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for the 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_IDX = 1000024
    MAX_N = 1000000
    MIN_N = 1000
    
    # Precompute smooth parts using NumPy for speed
    smooth = np.ones(MAX_IDX + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, extract the full power of p
        # We need to find for each m, the highest power of p dividing m
        # Then multiply smooth[m] by that power
        
        # Efficient approach: for each power p^k, mark multiples
        # but we need the FULL power, not just whether divisible
        
        # Alternative: use integer division approach
        # For m in range(p, MAX_IDX+1, p):
        #   count how many times p divides m
        #   smooth[m] *= p ** count
        
        # Vectorized approach using np.where and integer operations
        # For p=2: we can use bit operations, but let's do a general approach
        
        # Process in chunks to avoid memory issues
        # Actually, for N=1M, a simple loop with NumPy operations per prime should be fine
        
        # For each prime, process its multiples
        multiples = np.arange(p, MAX_IDX + 1, p, dtype=np.int64)
        if len(multiples) == 0:
            continue
        
        # For each multiple, extract the full power of p
        # This is tricky to vectorize directly, so we do it in a loop
        # but only over the multiples (N/p of them)
        
        # For small p like 2, N/p is ~500K which is too slow in Python
        # Let's use a different approach: iterative power extraction
        
        # For p=2: use bit_length and masking
        # For general p: we can use repeated division
        
        # Actually, let me use a smarter approach:
        # Precompute for each m, the p-smooth part by iterating primes
        # and for each prime, dividing out all factors
        
        # The original approach is actually fine for N=1M
        # Let's just do it in Python but optimize the inner loop
        
        pass  # We'll do the sieve in pure Python but optimized
    
    # Do the sieve in pure Python - it's fast enough for N=1M
    smooth_list = [1] * (MAX_IDX + 1)
    
    for p in primes:
        for mult in range(p, MAX_IDX + 1, p):
            val = mult
            while val % p == 0:
                smooth_list[mult] *= p
                val //= p
    
    # Convert to numpy for faster sliding window? 
    # Actually, Python list with defaultdict is fine for sliding window
    
    # Sliding window
    window_counts = defaultdict(int)
    for i in range(MIN_N + 1, MIN_N + 25):
        window_counts[smooth_list[i]] += 1
        
    min_distinct = len(window_counts)
    best_n = MIN_N
    
    if min_distinct == 1:
        return best_n
    
    # Slide the window
    for n in range(MIN_N + 1, MAX_N + 1):
        if time.time() > deadline:
            break
            
        val_out = smooth_list[n]
        window_counts[val_out] -= 1
        if window_counts[val_out] == 0:
            del window_counts[val_out]
            
        val_in = smooth_list[n + 24]
        window_counts[val_in] = window_counts.get(val_in, 0) + 1
        
        current_distinct = len(window_counts)
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            if min_distinct == 1:
                break
                
    return best_n
