"""Construct the maximum pairwise noncoprime subset of [1,255] containing 255."""

def construct():
    # The set of multiples of 3 in [1, 255] is a valid pairwise noncoprime set
    # containing 255. Its size is floor(255/3) = 85.
    # This is the maximum possible cardinality for this problem.
    return [x for x in range(1, 256) if x % 3 == 0]
