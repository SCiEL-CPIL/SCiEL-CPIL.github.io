"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    """
    target_divisor = 120
    
    # Use a sieve to compute phi(n) for all n up to a reasonable limit.
    # The smallest n with phi(n) divisible by 120 is 143, which is small.
    # We'll compute up to 10000 to be safe, which is very fast.
    
    limit = 10000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n with phi(n) divisible by 120
    for n in range(1, limit + 1):
        if phi[n] % target_divisor == 0:
            return n
    
    # Fallback: return a known valid answer
    return 143
# EVOLVE-BLOCK-END
