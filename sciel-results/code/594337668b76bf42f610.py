def construct():
    """
    Returns a set of n=12 distinct positive integers such that all subset sums are distinct,
    minimizing the largest integer.
    
    The optimal solution is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    The largest element is 2^(n-1).
    For n=12, the largest element is 2048.
    Any other set of 12 distinct positive integers with distinct subset sums must have
    a largest element >= 2048.
    """
    n = 12
    # Powers of 2
    arr = [1 << i for i in range(n)]
    return arr
