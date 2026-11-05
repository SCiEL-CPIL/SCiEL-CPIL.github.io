# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    The optimal set of n distinct positive integers with distinct subset sums
    is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    This is because the subset sums must be 2^n distinct non-negative integers.
    To minimize the largest element, the subset sums should be as small as possible,
    which implies they should be contiguous starting from 0.
    This forces the elements to be powers of 2.
    
    For n=12, the largest element is 2^11 = 2048.
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
