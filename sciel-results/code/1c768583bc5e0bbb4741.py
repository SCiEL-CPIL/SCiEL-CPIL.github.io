"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The maximum number of multiplicatively independent integers in [2, 40]
    # is bounded by the number of distinct primes <= 40, which is 12.
    # These 12 primes are:
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Any subset of these has distinct products because they are linearly
    # independent in the vector space of prime exponents.
    # No set of size > 12 can exist in this range.
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# EVOLVE-BLOCK-END
