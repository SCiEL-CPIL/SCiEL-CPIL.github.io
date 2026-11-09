def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    # The set of all multiples of 3 in [1, 255] is a clique of size 85.
    # 255 is a multiple of 3, so it is included.
    # Any other element not divisible by 3 would require removing many elements
    # from the multiples of 3 to maintain the clique property, resulting in a smaller set.
    # Thus, the maximum cardinality is 85.
    return list(range(3, 256, 3))
