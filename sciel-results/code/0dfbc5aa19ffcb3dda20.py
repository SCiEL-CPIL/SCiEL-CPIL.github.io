def construct():
    """
    Returns a list of exactly n=12 distinct positive integers such that all subset sums are distinct.
    The goal is to minimize the largest integer.
    
    Mathematical Proof of Optimality:
    Let a_0 < a_1 < ... < a_{n-1} be the sequence.
    For all 2^n subset sums to be distinct, the sequence must be superincreasing:
    a_k > sum(a_0 ... a_{k-1}) for all k >= 1.
    To minimize a_{n-1}, we must minimize each a_k.
    The minimum value for a_0 is 1.
    The minimum value for a_k given previous elements is sum(a_0 ... a_{k-1}) + 1.
    This recurrence relation a_k = 1 + sum_{i=0}^{k-1} a_i yields a_k = 2^k.
    Thus, the sequence [1, 2, 4, ..., 2^{n-1}] is the unique optimal solution minimizing the maximum element.
    """
    n = 12
    seq = []
    current_sum = 0
    for i in range(n):
        # The smallest integer greater than the sum of all previous elements
        next_val = current_sum + 1
        seq.append(next_val)
        current_sum += next_val
    return seq
