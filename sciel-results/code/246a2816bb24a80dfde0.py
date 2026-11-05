"""Construct the maximum pairwise noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17
    # Any element in the set must share a factor with 255 (divisible by 3, 5, or 17).
    # For pairwise noncoprime: if we pick all multiples of 3, any two share factor 3.
    # |multiples of 3 in [1,255]| = floor(255/3) = 85.
    # This is the largest possible clique since 3 is the smallest prime factor of 255.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
