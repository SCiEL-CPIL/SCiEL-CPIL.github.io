"""
Constructs a set of 12 distinct positive integers with distinct subset sums,
minimizing the largest integer.
The theoretical minimum for the largest integer in such a set of size n is 2^(n-1).
This is achieved by the sequence 1, 2, 4, ..., 2^(n-1).
"""

def construct():
    n = 12
    return [1 << i for i in range(n)]
