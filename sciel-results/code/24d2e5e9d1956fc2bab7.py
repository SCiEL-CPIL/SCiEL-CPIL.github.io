"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 90s kill
    
    # Direct constructive search: find smallest n with phi(n) % 120 == 0
    # phi(n) = product of phi(p^k) for prime powers p^k in factorization of n
    # phi(p^k) = p^(k-1) * (p-1)
    # We need phi(n) divisible by 120 = 2^3 * 3 * 5
    
    # Strategy: iterate n from 1 upward, compute phi(n) quickly using factorization
    # But for efficiency, we can use a smarter approach:
    # Generate candidate n values by considering which prime powers contribute factors to phi
    
    # Actually, let's just compute phi(n) for small n directly until we find one
    # phi(n) for n up to ~200 should be fast enough
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def phi(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    # Search from n=1 upward
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
        if time.time() > deadline:
            break
    
    return 144  # fallback
# EVOLVE-BLOCK-END
