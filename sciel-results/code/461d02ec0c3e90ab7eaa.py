import time
import sys
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 123457
    
    # Primes to check
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate set of p-smooth numbers <= limit
    def get_smooth_numbers(primes_list, limit):
        smooth_set = {1}
        # We can use a BFS-like approach or just iterate
        # Since limit is small, we can just generate all combinations
        # But iterative expansion is safer
        
        # Start with 1
        current_smooths = [1]
        for p in primes_list:
            new_smooths = []
            for x in current_smooths:
                val = x
                while val * p <= limit:
                    val *= p
                    if val not in smooth_set:
                        new_smooths.append(val)
                        smooth_set.add(val)
            current_smooths.extend(new_smooths)
        return smooth_set

    for idx_p, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
            
        current_primes = primes[:idx_p + 1]
        
        # Generate smooth numbers
        smooth_set = get_smooth_numbers(current_primes, N)
        
        # Check for pair
        for a in smooth_set:
            b = N - a
            if 1 <= b <= N and b in smooth_set:
                return [a, b]
    
    # Fallback: Sieve LPF
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break
                
    return [best_a, best_b]
