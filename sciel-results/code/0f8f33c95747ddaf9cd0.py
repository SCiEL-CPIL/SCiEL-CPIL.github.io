# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute the smooth part (factor composed of primes < 24) for numbers up to 1000024
    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    smooth = [0] * (MAX_VAL + 1)
    
    # Initialize all to 1
    for i in range(1, MAX_VAL + 1):
        smooth[i] = i
    
    # For each prime < 24, divide out its powers
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            while smooth[multiple] % p == 0:
                smooth[multiple] //= p
    
    # Now smooth[x] = x with all factors of primes < 24 removed... wait, that's the coprime part.
    # We want the SMOOTH part, i.e., the part composed ONLY of primes < 24.
    # Let me redo: smooth_part[x] = product of p^a for p in primes where p^a || x
    
    smooth_part = [1] * (MAX_VAL + 1)
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            temp = multiple
            while temp % p == 0:
                smooth_part[multiple] *= p
                temp //= p
    
    # Now for each candidate n, compute distinct smooth parts of n+1..n+24
    best_n = 1000
    best_count = float('inf')
    
    # Check all candidates
    for n in range(1000, MAX_N - 24 + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Compute distinct smooth parts
        seen = set()
        for i in range(1, 25):
            seen.add(smooth_part[n + i])
        
        count = len(seen)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
