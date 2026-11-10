"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
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
    
    # Map distinct small_factor values to small integer IDs for fast distinct counting
    # This allows us to use a counting approach in the sliding window
    unique_vals = np.unique(small_factor)
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    num_ids = len(unique_vals)
    
    # Create ID array
    sf_ids = np.empty(max_val + 1, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        sf_ids[sf_ids == 0]  # placeholder, will use vectorized mapping below
    
    # Vectorized mapping: use searchsorted on sorted unique values
    sorted_unique = np.sort(unique_vals)
    sf_ids = np.searchsorted(sorted_unique, small_factor).astype(np.int32)
    
    # Now slide window of 24 over n in [1000, 1000000]
    # For candidate n, the numbers are n+1, n+2, ..., n+24
    # We need distinct count of sf_ids[n+1 : n+25]
    
    best_score = 25  # worst case: all distinct
    best_n = 1000
    
    # Sliding window distinct count using a frequency array
    # Window size = 24, IDs range [0, num_ids)
    window_size = 24
    
    # Initialize frequency for the first window (n=1000: indices 1001..10024)
    freq = np.zeros(num_ids, dtype=np.int32)
    for i in range(1001, 1001 + window_size):
        freq[sf_ids[i]] += 1
    
    current_distinct = np.count_nonzero(freq)
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score == 1:
            return best_n
    
    # Slide the window
    for n in range(1001, max_n + 1):
        # Remove element at position n (leaving the window)
        freq[sf_ids[n]] -= 1
        # Add element at position n + 24 (entering the window)
        freq[sf_ids[n + window_size]] += 1
        
        # Update distinct count
        if freq[sf_ids[n]] == 0:
            current_distinct -= 1
        if freq[sf_ids[n + window_size]] == 1:
            current_distinct += 1
        
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
            if best_score == 1:
                break
        
        # Check time periodically
        if n % 10000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
