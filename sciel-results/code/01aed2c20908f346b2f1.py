def construct():
    """
    Return the maximum-size subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b in the set.
    
    255 = 3 * 5 * 17. The set of all multiples of 3 in [1, 255] has size
    floor(255/3) = 85 and is pairwise non-coprime (all share factor 3).
    This is optimal: any valid set containing 255 must have every element
    share a factor with 255 (i.e., be divisible by 3, 5, or 17). But the
    element 3 is in the multiples-of-3 set and gcd(3, x) > 1 requires x
    divisible by 3. So no element outside the multiples of 3 can be added
    to a set that includes 3. Since 3 must be in any maximal set (it's
    compatible with 255 and with all other multiples of 3), the maximum
    size is exactly 85.
    """
    return list(range(3, 256, 3))
