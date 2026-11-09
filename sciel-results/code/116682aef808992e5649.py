"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _totient(n):
    """Compute Euler's totient phi(n) via trial division."""
    if n == 1:
        return 1
    result = n
    x = n
    # Factor out 2
    if x % 2 == 0:
        result -= result // 2
        while x % 2 == 0:
            x //= 2
    # Factor out odd primes
    p = 3
    while p * p <= x:
        if x % p == 0:
            result -= result // p
            while x % p == 0:
                x //= p
        p += 2
    # If x is still greater than 1, it's a prime factor
    if x > 1:
        result -= result // x
    return result

def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    deadline = time.time() + 55  # safety margin before 60s limit
    for n in range(1, 1000001):
        if time.time() > deadline:
            # Fallback: return a known valid answer
            # phi(120*120) is divisible by 120 (from original program)
            return 120 * 120
        phi_n = _totient(n)
        if phi_n % 120 == 0:
            return n
    # Should not reach here, but fallback
    return 120 * 120
# EVOLVE-BLOCK-END
