import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # For each smoothness bound y, generate all y-smooth numbers in [N_MIN, N_MAX]
    # A number is y-smooth if all its prime factors are <= y
    # We want to find the smallest y such that there exist consecutive y-smooth numbers in range
    
    # First, let's try to find if any consecutive pair has max lpf <= some small value
    # Strategy: generate smooth numbers for increasing y, check for consecutive pairs
    
    def generate_smooth_numbers(limit, max_val, primes):
        """Generate all numbers <= max_val with all prime factors in primes list, >= limit"""
        results = set()
        n_primes = len(primes)
        
        def gen(idx, current):
            if current > max_val:
                return
            if current >= limit:
                results.add(current)
            if idx >= n_primes:
                return
            p = primes[idx]
            # Multiply by p^k
            val = current
            while val <= max_val:
                gen(idx + 1, val)
                if val > max_val // p:
                    break
                val *= p
        
        gen(0, 1)
        return results
    
    # Primes to consider for smoothness
    prime_bounds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # For each smoothness bound, check if consecutive smooth numbers exist in range
    for y in prime_bounds:
        if time.time() > deadline:
            break
        
        # Get all primes <= y
        if y == 2:
            primes = [2]
        elif y == 3:
            primes = [2, 3]
        elif y == 5:
            primes = [2, 3, 5]
        elif y == 7:
            primes = [2, 3, 5, 7]
        elif y == 11:
            primes = [2, 3, 5, 7, 11]
        elif y == 13:
            primes = [2, 3, 5, 7, 11, 13]
        elif y == 17:
            primes = [2, 3, 5, 7, 11, 13, 17]
        elif y == 19:
            primes = [2, 3, 5, 7, 11, 13, 17, 19]
        elif y == 23:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        elif y == 29:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        elif y == 31:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        elif y == 37:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        elif y == 41:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        elif y == 43:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        elif y == 47:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        smooth = generate_smooth_numbers(N_MIN, N_MAX, primes)
        
        if not smooth:
            continue
        
        # Sort and check for consecutive pairs
        sorted_smooth = sorted(smooth)
        consecutive_pairs = []
        for i in range(len(sorted_smooth) - 1):
            if sorted_smooth[i + 1] - sorted_smooth[i] == 1:
                consecutive_pairs.append(sorted_smooth[i])
        
        if consecutive_pairs:
            # Found consecutive y-smooth numbers; the largest prime factor is at most y
            # Return the first such n (smallest n for this y)
            return int(consecutive_pairs[0])
    
    # Fallback: use sieve to compute LPF for all numbers and find minimum
    # This should rarely be needed
    N_SIEVE = N_MAX + 2
    lpf = np.zeros(N_SIEVE, dtype=np.int64)
    
    for i in range(2, N_SIEVE):
        if lpf[i] == 0:
            lpf[i::i] = i
        if time.time() > deadline:
            break
    
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_np1 = lpf[N_MIN+1:N_MAX+2]
    obj_vals = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(obj_vals)
    return int(N_MIN + min_idx)
