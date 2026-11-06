"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    
    # We need small parts for numbers up to MAX_N + WINDOW_SIZE
    M = MAX_N + WINDOW_SIZE + 10
    
    # Precompute small parts using numpy vectorized sieve
    # sp[x] = product of all prime factors < 24 of x
    sp = np.ones(M + 1, dtype=np.int64)
    
    for p in primes:
        # Find all multiples of p
        multiples = np.arange(p, M + 1, p)
        # For each multiple, extract all factors of p
        # We need to do this iteratively since we can't vectorize the "while divisible" easily
        # But we can do: for each power p^k, mark multiples of p^k
        power = p
        while power <= M:
            multiples_power = np.arange(power, M + 1, power)
            sp[multiples_power] *= p
            power *= p
            if power > M:
                break
    
    # Convert to list for faster indexing in the sliding window
    sp_list = sp.tolist()
    
    # Sliding window to find min distinct small parts
    counts = {}
    distinct = 0
    
    # Initialize counts for the first window
    for i in range(1, WINDOW_SIZE + 1):
        v = sp_list[MIN_N + i]
        c = counts.get(v, 0)
        if c == 0:
            distinct += 1
        counts[v] = c + 1
        
    best_n = MIN_N
    best_distinct = distinct
    
    # Slide the window
    for n in range(MIN_N, MAX_N + 1):
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            
        # Check time every 5000 steps
        if (n - MIN_N) % 5000 == 0:
            if time.time() > deadline:
                break
        
        if n < MAX_N:
            v_out = sp_list[n + 1]
            c_out = counts[v_out]
            if c_out == 1:
                distinct -= 1
                del counts[v_out]
            else:
                counts[v_out] = c_out - 1
            
            v_in = sp_list[n + 1 + WINDOW_SIZE]
            c_in = counts.get(v_in, 0)
            if c_in == 0:
                distinct += 1
            counts[v_in] = c_in + 1
            
    return best_n
# EVOLVE-BLOCK-END
