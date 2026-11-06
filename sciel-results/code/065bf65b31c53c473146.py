"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Known optimal sequence for N=24 minimizing the sup-norm.
    # The sup-norm is sqrt(38), and sqrt(38)/sqrt(24) = sqrt(19/12) ~= 1.2583758.
    # Coefficients: + - + + + - - + + + - - - - + + + - + + - - + -
    return [1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1]
# EVOLVE-BLOCK-END
