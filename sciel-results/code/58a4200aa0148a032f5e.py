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

import math
import time


def construct():
    deadline = time.time() + 58  # 58s safety margin under 60s limit

    n = 255
    # Prime factors of 255: 3, 5, 17
    # Smallest prime factor gives the largest group of multiples
    p = 3

    # All multiples of p in [1, 255]
    result = list(range(p, n + 1, p))

    # Verify 255 is included
    assert 255 in result, "255 must be in the result"

    # Verify pairwise gcd > 1 (fast check: all elements share factor p)
    for x in result:
        assert x % p == 0, f"{x} not divisible by {p}"

    # Fallback (should never trigger)
    if time.time() > deadline:
        return [n]

    return result
