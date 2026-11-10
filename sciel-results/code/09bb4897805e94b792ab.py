"""
Solves the problem of finding 12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

The optimal solution is the sequence of powers of 2: 1, 2, 4, ..., 2^11.
This sequence is superincreasing, ensuring all 2^12 subset sums are distinct.
The largest element is 2^11 = 2048.
It can be proven that no set of 12 distinct positive integers with distinct subset sums
can have a largest element smaller than 2^11.
"""

def construct():
    # n = 12
    # The minimal largest element for n distinct subset sums is 2^(n-1).
    # We return the powers of 2 from 2^0 to 2^11.
    return [1 << i for i in range(12)]
