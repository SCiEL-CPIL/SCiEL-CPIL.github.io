"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi for all numbers up to 1000000, then scans for the answer.
    """
    deadline = time.time() + 80  # Safety margin before 90s kill
    N = 1000000
    
    # Sieve of Eratosthenes style for computing phi
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
        
        # Check if we're still within time budget
        if time.time() > deadline:
            break
    
    # Scan for the smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (shouldn't reach here)
    return N
# EVOLVE-BLOCK-END
