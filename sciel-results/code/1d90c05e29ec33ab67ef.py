"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    
    255 = 3 * 5 * 17.
    
    Any valid set containing 255 must have every element sharing a prime factor
    with 255 (i.e., divisible by 3, 5, or 17).
    
    To maximize cardinality, we want the largest possible clique.
    The set of all multiples of 3 in [1, 255] has size 85.
    The set of all multiples of 5 in [1, 255] has size 51.
    The set of all multiples of 17 in [1, 255] has size 15.
    
    Can we do better than 85 by combining elements from different prime groups?
    If we include an element divisible by 5 but not 3, it must share a factor > 1
    with every other element. If we also have elements divisible by 3 but not 5,
    their gcd would be 1 unless they share another factor (17).
    
    However, the key insight is: if we take ALL multiples of 3, that's 85 elements.
    Could adding any element from the other groups help? An element divisible by 5
    but not 3 (e.g., 5, 10, 20, ...) would need to share a factor with all 85 multiples
    of 3. But gcd(5, 3) = 1, so 5 can't be added. Similarly for 10: gcd(10, 3) = 1.
    
    What about elements divisible by 15 (both 3 and 5)? Those are already in the
    multiples of 3 set.
    
    What about elements divisible by 17 but not 3? E.g., 17, 34, 68, ...
    gcd(17, 3) = 1, so 17 can't be added to the multiples-of-3 set.
    
    So the maximum is 85, achieved by taking all multiples of 3.
    """
    # All multiples of 3 in [1, 255]
    # 255 / 3 = 85, so multiples are 3, 6, 9, ..., 255
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
