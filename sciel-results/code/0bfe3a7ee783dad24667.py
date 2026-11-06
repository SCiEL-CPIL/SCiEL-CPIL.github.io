import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s
    
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # we need smooth parts up to 1000024
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth part for each number using vectorized sieve
    smooth = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            smooth[multiples[mask]] *= p
            temp[mask] //= p
    
    # Sliding window with frequency counter for O(1) per step
    # Window for n: indices n+1 to n+24
    freq = defaultdict(int)
    distinct_count = 0
    for i in range(1001, 1001 + WINDOW):
        v = smooth[i]
        if freq[v] == 0:
            distinct_count += 1
        freq[v] += 1
    
    best_n = 1000
    best_count = distinct_count
    
    # Slide window from n=1001 to MAX_N
    for n in range(1001, MAX_N + 1):
        if time.time() - start_time > deadline:
            break
        
        # Transition from n-1 to n: remove index n, add index n+24
        out_val = smooth[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            distinct_count -= 1
        
        in_val = smooth[n + WINDOW]
        if freq[in_val] == 0:
            distinct_count += 1
        freq[in_val] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                break
    
    return best_n
