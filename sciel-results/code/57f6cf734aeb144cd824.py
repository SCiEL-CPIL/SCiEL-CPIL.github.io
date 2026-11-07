import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 75  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Precompute smallest prime factor for all numbers up to MAX_N
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Cache for sigma values
    sigma_cache = {1: 1}
    
    def sigma(n):
        if n in sigma_cache:
            return sigma_cache[n]
        orig_n = n
        result = 1
        while n > 1:
            p = spf[n]
            power = 1
            while n % p == 0:
                n //= p
                power *= p
            result *= (power * p - 1) // (p - 1)
        sigma_cache[orig_n] = result
        return result
    
    # Search for a, b with sigma(a) = sigma(b) minimizing |a/b - 3/2|
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # For each small d (nonzero), iterate b from largest down
    d_values = []
    for d in range(1, 50):
        d_values.append(d)
        d_values.append(-d)
    
    for d in d_values:
        if time.time() - start_time > DEADLINE:
            break
        
        parity = d % 2
        b_min = max(1, math.ceil((2 - d) / 3))
        b_max = min(MAX_N, math.floor((2 * MAX_N - d) / 3))
        
        if b_max % 2 != parity:
            b_max -= 1
        if b_min % 2 != parity:
            b_min += 1
        
        if b_min > b_max:
            continue
        
        b = b_max
        while b >= b_min:
            if time.time() - start_time > DEADLINE:
                break
            
            a = (3 * b + d) // 2
            if a >= 1 and a <= MAX_N:
                sa = sigma(a)
                sb = sigma(b)
                if sa == sb:
                    error = abs(d) / (2 * b)
                    if error < best_error:
                        best_error = error
                        best_a = a
                        best_b = b
            
            b -= 2
    
    if best_a is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
