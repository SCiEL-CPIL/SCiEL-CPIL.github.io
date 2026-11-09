"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17
    # The largest pairwise non-coprime subset containing 255
    # is formed by all multiples of the smallest prime factor (3).
    # This gives 85 elements: 3, 6, ..., 255.
    return [i for i in range(3, 256, 3)]
# EVOLVE-BLOCK-END
