"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small factor for all numbers up to 1000024
    max_val = 1000024
    small_factor = [1] * (max_val + 1)
    
    for m in range(1, max_val + 1):
        temp = m
        factor = 1
        for p in primes:
            if temp == 1:
                break
            while temp % p == 0:
                factor *= p
                temp //= p
        small_factor[m] = factor
    
    # Sliding window to find minimum distinct factors
    best_n = 1000
    best_score = 25  # max possible is 24 distinct factors
    
    factor_count = defaultdict(int)
    distinct = 0
    
    # Initialize window for n=1000: factors of 1001 to 1024
    for i in range(1, 25):
        f = small_factor[1000 + i]
        factor_count[f] += 1
        if factor_count[f] == 1:
            distinct += 1
    
    best_score = distinct
    best_n = 1000
    
    # Slide the window from n=1001 to n=1000000
    for n in range(1001, 1000001):
        # Remove element at index n (leftmost of previous window)
        f_remove = small_factor[n]
        factor_count[f_remove] -= 1
        if factor_count[f_remove] == 0:
            distinct -= 1
        
        # Add element at index n+24 (new rightmost)
        f_add = small_factor[n + 24]
        factor_count[f_add] += 1
        if factor_count[f_add] == 1:
            distinct += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
        
        if best_score <= 1:
            break
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
