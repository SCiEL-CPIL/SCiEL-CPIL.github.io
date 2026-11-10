"""Rewritten program: maximize pairwise-noncoprime subset of [1, 255] containing 255.

Strategy: 255 = 3 * 5 * 17. Take all multiples of the smallest prime factor (3).
Every pair shares factor 3, and 255 is included. Cardinality = floor(255/3) = 85,
which beats the seed's 51 (multiples of 5).
"""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17; use the smallest prime factor 3 to maximize count.
    # All multiples of 3 in [1, 255] form a pairwise-noncoprime set containing 255.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
