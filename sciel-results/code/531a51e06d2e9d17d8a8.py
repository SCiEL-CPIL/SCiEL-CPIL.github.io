"""
Rewritten program to maximize the cardinality of a pairwise non-coprime subset of [1, 255] containing 255.

Analysis:
255 = 3 * 5 * 17.
To maximize the size of a set S such that gcd(a, b) > 1 for all a, b in S,
we should choose the prime factor p of 255 that yields the largest number of multiples in [1, 255].
The number of multiples of p is floor(255 / p).
- p = 3: floor(255 / 3) = 85
- p = 5: floor(255 / 5) = 51
- p = 17: floor(255 / 17) = 15

The largest set is the multiples of 3.
This set contains 255 (since 255 is divisible by 3).
Any two distinct multiples of 3 have gcd at least 3, so the condition is satisfied.
It can be proven that no larger set exists for n=255.
"""
# EVOLVE-BLOCK-START
def construct():
    # Return all multiples of 3 in the range [1, 255]
    # This includes 255 and ensures gcd(a, b) >= 3 for all pairs.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
