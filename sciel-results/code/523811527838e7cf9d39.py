"""
Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

Strategy: 255 = 3 * 5 * 17.
The set of all multiples of the smallest prime factor (3) in [1, 255]
forms a pairwise-noncoprime set containing 255.
Cardinality = floor(255/3) = 85.
This is the maximum possible size for such a set.
"""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17.
    # The smallest prime factor is 3.
    # All multiples of 3 in [1, 255] share the factor 3, so gcd(a,b) >= 3 > 1.
    # 255 is a multiple of 3, so it is included.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
