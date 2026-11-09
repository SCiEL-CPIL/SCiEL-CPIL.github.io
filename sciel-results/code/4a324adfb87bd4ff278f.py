"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute small-prime factors for all numbers in [1001, 1000024]
    min_val = 1001
    max_val = 1000024
    size = max_val - min_val + 1
    
    # Sieve-based precomputation: for each prime, process its multiples
    factors = [1] * (size + 1)  # 1-indexed
    for p in _SMALL_PRIMES:
        start = ((min_val + p - 1) // p) * p
        for m in range(start, max_val + 1, p):
            idx = m - min_val + 1
            while m % p == 0:
                factors[idx] *= p
                m //= p
    
    # Sliding window to find minimum distinct count
    best_n = 1000
    best_val = float('inf')
    
    freq = defaultdict(int)
    distinct_count = 0
    
    for i in range(24):
        f = factors[i + 1]
        if freq[f] == 0:
            distinct_count += 1
        freq[f] += 1
    
    best_val = distinct_count
    best_n = 1000
    
    for s in range(1, 999001):
        out_f = factors[s]
        freq[out_f] -= 1
        if freq[out_f] == 0:
            distinct_count -= 1
            del freq[out_f]
        
        in_f = factors[s + 24]
        if freq[in_f] == 0:
            distinct_count += 1
        freq[in_f] += 1
        
        if distinct_count < best_val:
            best_val = distinct_count
            best_n = 1000 + s
    
    return best_n
# EVOLVE-BLOCK-END
