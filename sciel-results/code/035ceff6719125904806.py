"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

def construct():
    """
    Find n in [1000, 1000000] such that for each n+i (1<=i<=24),
    the factor composed of primes strictly below 24 has the minimum
    number of distinct prime factors.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    
    In any 24 consecutive integers, every prime p < 24 divides at least
    one of them (since 24 > 23 >= p for all such p). Therefore, the
    small-prime factor of at least one n+i will include each prime < 24,
    making the count of distinct primes always 9 for any valid n.
    
    Since the score is constant, we return a valid n immediately.
    """
    return 1000
# EVOLVE-BLOCK-END
