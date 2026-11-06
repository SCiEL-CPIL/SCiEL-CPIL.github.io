import time
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 1000000
    max_val = N + 24
    sqrt_max = int(max_val**0.5) + 1
    
    # Compute 23-smooth part
    smooth = np.arange(max_val + 1, dtype=np.int64)
    
    # Sieve for primes up to sqrt_max
    is_prime_small = np.ones(sqrt_max + 1, dtype=bool)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(sqrt_max**0.5) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = False
    
    # Get primes >= 29 and <= sqrt_max
    small_primes = [p for p in range(29, sqrt_max + 1) if is_prime_small[p]]
    
    # For each small prime p >= 29, divide out p from multiples
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            while smooth[m] % p == 0:
                smooth[m] //= p
    
    # Find best n
    best_n = 1000
    best_count = float('inf')
    
    smooth_list = smooth.tolist()
    
    freq = Counter()
    for i in range(1, 25):
        freq[smooth_list[1000 + i]] += 1
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
        
        val_remove = smooth_list[n]
        freq[val_remove] -= 1
        if freq[val_remove] == 0:
            del freq[val_remove]
        
        val_add = smooth_list[n + 24]
        freq[val_add] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
