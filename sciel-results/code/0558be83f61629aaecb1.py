"""
Returns the maximum cardinality subset of [1, 255] containing 255 
such that gcd(a, b) > 1 for all distinct pairs.

The prime factorization of 255 is 3 * 5 * 17.
To maximize the set size while maintaining the pairwise non-coprime property
and including 255, we should select all multiples of the smallest prime factor,
which is 3. This ensures that any two elements in the set share the factor 3,
thus having a gcd > 1. The size of this set is floor(255 / 3) = 85.
Any attempt to include numbers not divisible by 3 would require excluding
a larger number of multiples of 3 to maintain the pairwise constraint with
the remaining elements, resulting in a smaller set.
"""

def construct():
    return list(range(3, 256, 3))
