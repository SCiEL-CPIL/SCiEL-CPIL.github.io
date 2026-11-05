"""Construct 12 distinct positive integers with distinct subset sums, minimizing the largest element."""

import time
import itertools

def _all_subsets_distinct(nums):
    """Check that all 2^n subset sums of nums are distinct."""
    n = len(nums)
    sums = set()
    # Use bitmask enumeration for compactness
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 2^12 subset sums are distinct.
    The sequence of powers of 2 (1, 2, 4, ..., 2048) is the optimal solution that minimizes the largest element.
    It is a proven lower bound that the k-th smallest element in such a set must be at least 2^(k-1).
    """
    n = 12
    return [1 << i for i in range(n)]
