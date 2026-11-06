"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    # 255 = 3 * 5 * 17
    # The largest pairwise non-coprime subset of [1, 255] containing 255
    # is the set of all multiples of the smallest prime factor of 255, which is 3.
    # Size: 255 // 3 = 85.
    # This set is valid: gcd(3a, 3b) >= 3 > 1 for all distinct a, b.
    # It contains 255 since 255 = 3 * 85.
    # No larger valid set exists because any number not divisible by 3
    # would need to be paired only with numbers sharing its prime factors,
    # but this would force exclusion of many multiples of 3 that are
    # coprime to it, yielding a smaller total set.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
