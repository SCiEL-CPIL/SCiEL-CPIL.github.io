import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    # Max value we need: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        # Sieve-style: for each power p^k, multiply extracted by p at each multiple
        power = p
        while power <= MAX_VAL:
            extracted[power::power] *= p
            power *= p

    # Sliding window search
    # We want to minimize the number of distinct values in extracted[n+1 : n+WINDOW+1]
    
    # The values in extracted are <= MAX_VAL, so we can use an array for frequencies
    freq = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    distinct_count = 0
    best_count = WINDOW + 1
    best_n = N_MIN

    # Initialize window for n = N_MIN
    # Window covers extracted[N_MIN+1] to extracted[N_MIN+WINDOW]
    for i in range(1, WINDOW + 1):
        val = extracted[N_MIN + i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1

    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN

    # Slide window from N_MIN + 1 to N_MAX
    # Window for n is extracted[n+1 ... n+WINDOW]
    # Window for n+1 is extracted[n+2 ... n+WINDOW+1]
    # Remove extracted[n+1], Add extracted[n+WINDOW+1]
    
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove element at index n+1? 
        # Wait, window for n-1 was (n-1)+1 ... (n-1)+WINDOW = n ... n+WINDOW-1
        # Window for n is n+1 ... n+WINDOW
        # So we remove extracted[n] and add extracted[n+WINDOW]
        
        # Let's re-verify indices.
        # n=1000: indices 1001..1024
        # n=1001: indices 1002..1025
        # Remove: extracted[1001] (which is n for n=1001? No, 1001 is the new start-1? No.)
        # Prev window (n-1): (n-1)+1 to (n-1)+24 => n to n+23
        # Curr window (n): n+1 to n+24
        # Removed: extracted[n]
        # Added: extracted[n+24]
        
        val_out = extracted[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
            
        val_in = extracted[n + WINDOW]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                return best_n

    return best_n
