"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    # The theoretical minimum for the largest element in a set of n distinct
    # positive integers with distinct subset sums is 2^(n-1).
    # This is achieved uniquely (up to permutation) by the powers of 2.
    # n=12 -> largest element = 2^11 = 2048.
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
