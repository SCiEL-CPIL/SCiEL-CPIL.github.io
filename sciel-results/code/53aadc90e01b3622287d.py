"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def _is_complete(arr):
    """Check if all subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use bit manipulation to generate all 2^n subset sums
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct,
    minimizing the largest integer.
    
    The minimum largest element for n=12 is 2^11 = 2048, achieved by the powers of 2.
    """
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
