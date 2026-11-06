def construct():
    """
    Returns a set of 12 distinct positive integers with distinct subset sums.
    The optimal solution minimizes the largest element, which is 2^(12-1) = 2048.
    The powers of 2 form a superincreasing sequence, ensuring all subset sums are unique.
    """
    return [1 << i for i in range(12)]
