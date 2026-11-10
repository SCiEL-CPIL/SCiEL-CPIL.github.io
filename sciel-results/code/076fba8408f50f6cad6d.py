def construct():
    # 255 = 3 * 5 * 17. The largest pairwise-noncoprime subset of [1,255]
    # containing 255 is all multiples of the smallest prime factor (3).
    # Any element not divisible by 3 would be coprime to some multiple of 3
    # (e.g., 3 itself if the element isn't divisible by 3), so we cannot
    # add any non-multiple-of-3 to the set of all multiples of 3.
    # Cardinality = floor(255/3) = 85.
    return list(range(3, 256, 3))
