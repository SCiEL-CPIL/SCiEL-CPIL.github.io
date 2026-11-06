"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 55.0

    # Compute phi(n) for all n up to a limit using a sieve
    limit = 50000
    phi = list(range(limit + 1))
    
    # Sieve for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback: 241 is prime, phi(241) = 240 = 120 * 2
    return 241
# EVOLVE-BLOCK-END
