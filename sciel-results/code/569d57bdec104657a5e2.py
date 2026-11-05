def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    
    The set of all multiples of 3 in [1, 255] has size 85.
    255 is a multiple of 3.
    Any two distinct multiples of 3 have gcd at least 3.
    This is the maximum possible size because any valid set
    containing 255 must have elements sharing factors with 255 (3, 5, 17).
    The largest prime factor group is 3.
    """
    return list(range(3, 256, 3))
