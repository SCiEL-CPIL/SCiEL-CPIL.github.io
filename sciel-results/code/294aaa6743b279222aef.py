import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Vectorized sieve: for each prime p and each power p^k,
    # multiply small_part at indices divisible by p^k by p.
    # This correctly accumulates p^(v_p(n)) for each n.
    small_part = np.ones(size + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= size:
            small_part[power::power] *= p
            power *= p
    
    # Convert to Python list once for faster access in the sliding window
    sp = small_part.tolist()
    
    deadline = start_time + timeout - 5.0
    
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: indices 1001 to 1024
    freq = {}
    for i in range(1001, 1025):
        val = sp[i]
        freq[val] = freq.get(val, 0) + 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    if best_count == 1:
        return best_n
    
    # Slide window from n=1001 to max_n
    # Use local variables for speed
    sp_local = sp
    freq_local = freq
    distinct_local = distinct
    best_count_local = best_count
    best_n_local = best_n
    
    for n in range(1001, max_n + 1):
        # Remove element at position n
        out_val = sp_local[n]
        c = freq_local[out_val] - 1
        if c == 0:
            del freq_local[out_val]
            distinct_local -= 1
        else:
            freq_local[out_val] = c
        
        # Add element at position n+24
        in_val = sp_local[n + 24]
        c = freq_local.get(in_val, 0) + 1
        freq_local[in_val] = c
        if c == 1:
            distinct_local += 1
        
        if distinct_local < best_count_local:
            best_count_local = distinct_local
            best_n_local = n
            if best_count_local == 1:
                break
        
        if n % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n_local
# EVOLVE-BLOCK-END
