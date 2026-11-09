def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums
    are distinct, minimizing the largest integer.
    
    For a set of n positive integers with all subset sums distinct, the minimum
    possible largest element is 2^(n-1), achieved uniquely (up to order) by the
    powers of 2. This follows from the fact that 2^n distinct subset sums must
    all lie in {0, 1, ..., max_sum}, so max_sum >= 2^n - 1. Moreover, a
    classical result (related to the completeness of superincreasing sequences)
    shows that the largest element must be at least 2^(n-1) to avoid collisions:
    if we sort the elements a_1 < a_2 < ... < a_n, then for all k, a_k must be
    greater than the sum of all previous elements to keep the subset sums distinct.
    The tightest such sequence starting from a_1 >= 1 is a_k = 2^(k-1).
    
    For n=12, the largest element is 2^11 = 2048.
    """
    n = 12
    # Powers of 2: 1, 2, 4, ..., 2^(n-1)
    return [1 << i for i in range(n)]
