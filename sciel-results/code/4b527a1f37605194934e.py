"""Rewritten program: exact cardinality-indexed subset-sum DP to maximize set size."""
import sys
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers in 1..256 such that subset sums
    of different cardinalities are disjoint.
    Uses the set of powers of 2 up to 256.
    """
    # Powers of 2: 1, 2, 4, ..., 256
    # Size 9.
    # Validity: Sum of a subset of size k is an integer with exactly k bits set.
    # Thus, sums of different cardinalities are disjoint.
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

# EVOLVE-BLOCK-END
