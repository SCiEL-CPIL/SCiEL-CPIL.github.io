"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal set of 12 distinct positive integers with distinct subset sums.
    The set is the superincreasing sequence of powers of 2, which minimizes the largest element.
    """
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# EVOLVE-BLOCK-END
