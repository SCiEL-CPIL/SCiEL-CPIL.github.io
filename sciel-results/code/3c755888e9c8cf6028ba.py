"""
Optimal solution for distinct subset sums with minimal largest element.

The problem requires finding $n$ distinct positive integers such that all $2^n$ subset sums are distinct.
This property is equivalent to the sequence being superincreasing, i.e., $x_k > \sum_{i=1}^{k-1} x_i$.
To minimize the largest element $x_n$, we choose the smallest possible integer at each step:
$x_1 = 1$
$x_k = 1 + \sum_{i=1}^{k-1} x_i$

This recurrence yields $x_k = 2^{k-1}$.
For $n=12$, the sequence is $[1, 2, 4, \dots, 2048]$.
The largest element is $2048$, which is provably minimal.
"""

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 distinct positive integers with all subset sums distinct,
    minimizing the largest element.
    
    The optimal sequence is powers of 2: [1, 2, 4, ..., 2048].
    """
    n = 12
    seq = []
    current_sum = 0
    for i in range(n):
        # The minimal next element is 1 + sum of previous elements
        next_val = 1 + current_sum
        seq.append(next_val)
        current_sum += next_val
    return seq
# EVOLVE-BLOCK-END
