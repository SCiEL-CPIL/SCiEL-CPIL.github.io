def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    The optimal solution is the sequence of powers of 2: 1, 2, 4, ..., 2^11.
    This sequence is superincreasing (each element is greater than the sum of all previous elements),
    which is a necessary and sufficient condition for the subset sums to be distinct when the sequence is sorted in ascending order.
    Any other valid sequence must satisfy a_i > sum(a_j for j < i), which implies a_n >= 2^(n-1).
    For n=12, the minimum largest element is 2^11 = 2048.
    """
    return [1 << i for i in range(12)]
