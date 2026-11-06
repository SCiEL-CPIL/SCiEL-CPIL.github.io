"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The task requires a subset of [1, 255] containing 255,
    # such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    # We want to maximize the cardinality.
    
    # 255 = 3 * 5 * 17.
    # The smallest prime factor is 3.
    # Taking all multiples of 3 ensures that any two distinct elements
    # share a factor of 3, so their gcd is at least 3.
    # The size of this set is floor(255 / 3) = 85.
    
    # Can we do better?
    # If we try to include numbers not divisible by 3, they must share a factor
    # with all existing numbers. If we keep 3 in the set, any new number x
    # must have gcd(x, 3) > 1, implying 3 divides x. Thus, we cannot add
    # non-multiples of 3 if 3 is in the set.
    # If we remove 3, we might add non-multiples of 3, but we lose 1 element
    # and gain very few, because non-multiples of 3 must also share factors
    # with other small multiples of 3 (like 9, 27, etc.), which forces them
    # to be divisible by 3 anyway or requires removing many multiples of 3.
    # The set of multiples of the smallest prime factor is the standard optimal
    # solution for this class of problems.
    
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
