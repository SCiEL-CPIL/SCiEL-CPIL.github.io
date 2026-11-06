import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # Seconds, leave 5s margin
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    
    # 1. Compute Largest Prime Factor for all numbers up to 1000001
    # Step 1a: Generate primes using vectorized boolean sieve
    is_prime = np.ones(N_PLUS_1 + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N_PLUS_1)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Step 1b: Compute LPF by processing primes in increasing order
    # The last prime that divides a number is its largest prime factor
    lpf = np.zeros(N_PLUS_1 + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p
    
    # 2. Search for optimal n using vectorized operations
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    scores = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(scores)
    best_n = 10000 + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
