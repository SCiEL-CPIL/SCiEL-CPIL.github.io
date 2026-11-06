"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _phi(n):
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

def construct():
    # The smallest n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    # We verify this programmatically to ensure robustness.
    for n in range(1, 10000):
        if _phi(n) % 120 == 0:
            return n
    return 143
# EVOLVE-BLOCK-END
