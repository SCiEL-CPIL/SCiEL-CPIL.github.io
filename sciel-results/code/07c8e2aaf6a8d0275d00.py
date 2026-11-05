import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_small_primes = len(small_primes)
    
    # For each number m in [1, MAX_VAL], compute:
    # 1. The "small part" S[m]: the largest divisor of m composed only of primes < 24
    # 2. The "large part" L[m]: m / S[m]
    # The distinct factor count for a window is the number of distinct values of (S[m], L[m]) pairs
    # But actually, we just need distinct A[m] = S[m] values? No...
    # Wait, re-reading the task: "extract its complete factor composed of primes STRICTLY BELOW 24"
    # So for each n+i, we extract S[n+i] = product of all prime powers p^k where p < 24 divides n+i
    # Then we count distinct S values in the window.
    # So we need distinct S[m] values, not (S[m], L[m]) pairs.
    
    # Initialize S[m] = 1 for all m
    S = [1] * (MAX_VAL + 1)
    
    # For each small prime p, multiply S[m] by p^k for all m divisible by p
    for p in small_primes:
        # Find the highest power of p that could appear
        p_power = p
        while p_power * p <= MAX_VAL:
            p_power *= p
        
        # For each multiple of p, find the exact power and update S
        for m in range(p, MAX_VAL + 1, p):
            temp = m
            factor = 1
            while temp % p == 0:
                factor *= p
                temp //= p
            S[m] *= factor
    
    # Now S[m] is the complete factor of m composed of primes strictly below 24
    
    # Map distinct S values to IDs for fast comparison
    unique_S = {}
    S_ids = [0] * (MAX_VAL + 1)
    next_id = 0
    for m in range(1, MAX_VAL + 1):
        val = S[m]
        if val not in unique_S:
            unique_S[val] = next_id
            next_id += 1
        S_ids[m] = unique_S[val]
    
    # Search for n using sliding window
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Use a frequency array to count distinct IDs in the sliding window
    # First, check time
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Initialize window for n = N_MIN
    freq = [0] * next_id
    for i in range(N_MIN + 1, N_MIN + WINDOW + 1):
        freq[S_ids[i]] += 1
    
    distinct_count = sum(1 for f in freq if f > 0)
    
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = N_MIN
        if best_count == 1:
            return best_n
    
    # Slide the window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove n
        old_id = S_ids[n]
        freq[old_id] -= 1
        if freq[old_id] == 0:
            distinct_count -= 1
        
        # Add n + WINDOW
        new_id = S_ids[n + WINDOW]
        if freq[new_id] == 0:
            distinct_count += 1
        freq[new_id] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
    
    return best_n
