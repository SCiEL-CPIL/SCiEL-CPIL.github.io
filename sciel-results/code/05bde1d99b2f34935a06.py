import time
import math

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
    
    # Initialize
    extracted = [1] * (MAX_VAL + 1)
    
    # For each small prime, update extracted for its multiples
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            # Calculate the p-part of m
            factor = 1
            while val % p == 0:
                factor *= p
                val //= p
            extracted[m] *= factor

    # Now search for n that minimizes distinct extracted factors
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Optimization: If we find 1, we can break.
    # 1 is possible if all 24 numbers are coprime to the product of small primes.
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Calculate distinct count
        # Using a set comprehension or manual addition
        # Manual addition is often faster for small fixed sizes in CPython
        factors = set()
        # Unroll or just loop
        for i in range(1, WINDOW + 1):
            factors.add(extracted[n + i])
        
        count = len(factors)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
