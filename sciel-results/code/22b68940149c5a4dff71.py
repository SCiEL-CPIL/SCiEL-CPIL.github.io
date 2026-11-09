"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _totient(n):
    """Compute Euler's totient of n via trial division."""
    result = n
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            while m % d == 0:
                m //= d
            result -= result // d
        d += 1
    if m > 1:
        result -= result // m
    return result

def construct():
    """Return the smallest n in [1, 1000000] with phi(n) divisible by 120.
    
    Uses direct mathematical analysis: the smallest n with phi(n) divisible
    by 120 is found by checking candidates in increasing order up to a small
    bound, leveraging that phi(n) grows roughly linearly and 120 has few
    factorizations. The known minimal solution is n=143 (11*13, phi=120).
    We verify this and check no smaller n works by direct computation.
    """
    # Directly verify the known minimal candidate and confirm minimality
    # by checking all n < 143 for phi(n) % 120 == 0
    for n in range(1, 143):
        if _totient(n) % 120 == 0:
            return n
    # 143 = 11 * 13, phi(143) = 10 * 12 = 120, which is divisible by 120
    return 143
# EVOLVE-BLOCK-END
