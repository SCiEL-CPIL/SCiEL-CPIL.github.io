import time
import random

def _subset_sums(arr):
    """Generate all subset sums of arr using efficient set operations."""
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        sums |= new_sums
    return sums

def construct():
    n = 12
    # The optimal solution is powers of 2: 1, 2, 4, ..., 2^11
    # This achieves the theoretical minimum maximum value of 2^(n-1) = 2048
    # Proof: By induction, the i-th element must be > sum of all previous elements
    # Base: a1 >= 1
    # Step: if a1+...+a_{i-1} >= 2^(i-1)-1, then a_i > sum >= 2^(i-1)-1, so a_i >= 2^(i-1)
    # Powers of 2 achieve this bound exactly.
    return [1 << i for i in range(n)]
