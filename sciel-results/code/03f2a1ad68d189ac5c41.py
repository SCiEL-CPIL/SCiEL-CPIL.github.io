"""
Rewritten program to find the maximum set of distinct integers in 2..40
such that all subset products are distinct.
This is equivalent to finding a set of linearly independent vectors
in the prime exponent space. The dimension is the number of primes <= 40,
which is 12. The set of primes themselves is a valid basis.
"""

def construct():
    # Primes <= 40
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
