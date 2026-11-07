"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin out of 60s
    
    # Sieve of Eratosthenes for primes up to 256
    N = 256
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # The set of all primes is a valid solution.
    # Based on analysis, it is likely the maximum cardinality.
    return primes

# EVOLVE-BLOCK-END
