# EVOLVE-BLOCK-START
def construct():
    """
    Construct exactly 12 distinct positive Python integers <= 1048576
    such that every subset has a different sum. Minimize the largest integer.
    
    The optimal solution for distinct subset sums with minimal maximum element
    is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    The largest element is 2^(n-1).
    """
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
