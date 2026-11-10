"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to evaluate n in [1000, 1000000]
    # For each candidate n, look at n+1 .. n+24
    # Extract the factor composed of primes strictly below 24 from each
    # Count distinct extracted factors
    
    # Precompute small-prime factor for numbers up to 1000024
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Use numpy for fast precomputation
    # small_factor[x] = product of prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    # For each prime p < 24, multiply into all multiples of p
    for p in small_primes:
        # For each power of p: p, p^2, p^3, ...
        power = p
        while power <= max_val:
            # Multiply small_factor by p for all multiples of power
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
    
    # Convert to list for faster slicing in the window loop
    sf_list = small_factor.tolist()
    
    # Now slide window of 24 over n in [1000, 1000000]
    # For candidate n, the numbers are n+1, n+2, ..., n+24
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Precompute all window distinct counts using a sliding window approach
    # But first, let's just do the straightforward approach with numpy for speed
    
    # Create array of all small_factor values for positions 1001 to 1000024
    # For n, we need sf[n+1] through sf[n+24]
    # So for n=1000, we need sf[1001:1000+25] = sf[1001:10025]
    # For n=1000000, we need sf[1000001:1000000+25] = sf[1000001:1000025]
    
    # Let's use a different approach: compute distinct count for each window
    # Using numpy, we can vectorize some of this
    
    # Actually, let's just iterate but with numpy-backed operations
    # The key insight: we can precompute a "signature" for each window
    
    # For speed, let's try to find the minimum number of distinct factors
    # by checking windows where many values are equal
    
    # First, let's do a quick pass to find the minimum possible
    # We'll iterate through n and count distinct values in each window
    
    # Optimization: if best_score is already small, we can skip windows
    # that can't beat it
    
    # Let's use a smarter approach:
    # Group consecutive indices by their small_factor value
    # Then for each window of 24, count how many distinct groups it spans
    
    # Actually, the simplest fast approach in Python:
    # Use the fact that small_factor values are from a limited set
    # Let's just do the loop but with early termination
    
    for n in range(1000, max_n + 1):
        # Check time periodically
        if n % 5000 == 0:
            if time.time() > deadline:
                break
        
        # Extract factors for n+1 .. n+24
        window = sf_list[n + 1 : n + 25]
        
        # Count distinct values
        count = len(set(window))
        
        if count < best_score:
            best_score = count
            best_n = n
            if best_score <= 2:
                # Very good result, keep looking but with tighter deadline
                if best_score == 1:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
