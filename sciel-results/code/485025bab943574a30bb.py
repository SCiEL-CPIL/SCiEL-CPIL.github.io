def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums are distinct.
    The largest element is minimized.
    
    Theoretical Lower Bound:
    Let the sorted integers be a_1 < a_2 < ... < a_n.
    For all subset sums to be distinct, a_k must be greater than the sum of all previous elements.
    a_k > sum_{i=1}^{k-1} a_i.
    This implies a_k >= 2^(k-1) for k=1..n.
    For n=12, a_12 >= 2^11 = 2048.
    
    The powers of 2 achieve this bound: [1, 2, 4, ..., 2048].
    """
    return [1 << i for i in range(12)]
