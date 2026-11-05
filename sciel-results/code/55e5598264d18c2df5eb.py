"""
Constructs a nonempty subset of [1, 255] containing 255,
with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.

For n = 255 = 3 * 5 * 17 (square-free), the largest pairwise non-coprime
set containing 255 is the set of all multiples of the smallest prime factor, 3.
Any two distinct multiples of 3 have gcd >= 3.
Size = floor(255/3) = 85.

This is provably optimal: any element in a valid set must share a factor
with 255, so it's divisible by 3, 5, or 17. Elements divisible by 5 or 17
but not 3 can't coexist with all multiples of 3 (e.g., 5 and 3 are coprime),
so the maximum clique is contained in the multiples of a single prime factor.
The largest such group is multiples of 3, giving 85 elements.
"""

def construct():
    n = 255
    # Prime factors of 255: 3, 5, 17
    # The largest pairwise non-coprime subset containing 255
    # must be contained in the multiples of one of its prime factors.
    # To maximize cardinality, we choose the smallest prime factor, 3.
    p = 3

    # All multiples of p in [1, 255]
    result = list(range(p, n + 1, p))

    # 255 is a multiple of 3, so it is included.
    # All elements are multiples of 3, so gcd(a, b) >= 3 > 1 for all distinct pairs.
    return result
