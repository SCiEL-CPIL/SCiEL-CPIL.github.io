"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the largest subset of [1, 255] containing 255
    such that gcd(a, b) > 1 for all distinct pairs a, b in the set.
    
    The optimal strategy is to take all multiples of the smallest prime factor of 255.
    255 = 3 * 5 * 17. The smallest prime factor is 3.
    The set of multiples of 3 in [1, 255] has size floor(255/3) = 85.
    Any two multiples of 3 have gcd at least 3, which is > 1.
    255 is a multiple of 3, so it is included.
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
