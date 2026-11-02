"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set {0, 1, 3, 7, 15} is a prefix of the Mian-Chowla sequence.
    # It has distinct sums and differences, maximizing the ratio |U-U|/|U+U|
    # for its size while keeping max(U) relatively small.
    # For this set:
    # |U+U| = 15, |U-U| = 21, max(U) = 15
    # Score = 1 + log(21/15) / log(31) approx 1.098
    return [0, 1, 3, 7, 15]
# EVOLVE-BLOCK-END
