import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    LIMIT = 1000000
    MIN_N = 10000
    MAX_N = LIMIT
    
    # Generate primes up to a reasonable limit
    # We'll use trial division to find LPF for candidates
    def is_smooth(x, primes):
        """Check if x is smooth (all prime factors in primes list) and return LPF"""
        if x <= 1:
            return 0
        lpf = 1
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                lpf = p
                while x % p == 0:
                    x //= p
        if x > 1:
            lpf = x
        return lpf
    
    # Generate primes up to 2000 (should cover most cases)
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve_primes(2000)
    
    best_n = MIN_N
    best_val = float('inf')
    
    # Search for consecutive smooth pairs
    # Use segmented approach: check windows
    for n in range(MIN_N, MAX_N + 1):
        if time.time() > deadline - 2:
            break
        
        # Quick check: if n is even and large, it likely has a large prime factor
        # Compute LPF of n and n+1
        lpf_n = is_smooth(n, primes)
        if lpf_n >= best_val:
            continue
        
        lpf_np1 = is_smooth(n + 1, primes)
        max_lpf = max(lpf_n, lpf_np1)
        
        if max_lpf < best_val:
            best_val = max_lpf
            best_n = n
    
    return int(best_n)
