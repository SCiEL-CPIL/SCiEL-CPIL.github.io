def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The set of powers of 2 {1, 2, 4, ..., 2^11} is the unique optimal solution that minimizes
    the largest element. The largest element is 2^11 = 2048.
    """
    return [1 << i for i in range(12)]
