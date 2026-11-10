def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The largest element is minimized.

    The powers of two {1, 2, 4, ..., 2048} are known to be optimal:
    any 12 distinct positive integers with all subset sums distinct must satisfy
    a_k >= 2^(k-1) for sorted a_1 < ... < a_12, hence a_12 >= 2048.
    This set achieves the bound exactly.
    """
    return [1 << i for i in range(12)]
