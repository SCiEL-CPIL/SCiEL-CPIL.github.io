import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # Seconds, leave 5s margin
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    MAX_VAL = N_PLUS_1 + 1 # We need index up to 1000001
    
    # 1. Compute Largest Prime Factor for all numbers up to 1000001
    # Using NumPy for vectorized sieve updates
    # Initialize with 0. lpf[i] == 0 implies i is prime (or i < 2, which we ignore)
    lpf = np.zeros(MAX_VAL, dtype=np.int32)
    
    # Sieve of Eratosthenes variant for LPF
    # Iterate through potential primes
    # If lpf[i] is 0, i is prime
    for i in range(2, MAX_VAL):
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i
            # Since we process primes in increasing order, the last assignment
            # to lpf[j] will be from the largest prime factor of j.
            lpf[i::i] = i
            
    # 2. Search for optimal n using vectorized operations
    # We need n in [10000, 1000000]
    # Score is max(lpf[n], lpf[n+1])
    
    # Slices for lpf[n] and lpf[n+1]
    # n ranges from 10000 to 1000000 (inclusive)
    # lpf[n] indices: 10000 to 1000000 (inclusive) -> slice [10000:1000001]
    # lpf[n+1] indices: 100001 to 1000001 (inclusive) -> slice [10001:1000002]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Compute scores
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find index of minimum score
    min_idx = np.argmin(scores)
    
    # Calculate best_n
    best_n = 10000 + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
