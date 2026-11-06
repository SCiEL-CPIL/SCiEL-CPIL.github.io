"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal set of n distinct positive integers with distinct subset sums
    # minimizes the largest element.
    # The lower bound for the largest element a_n is ceil((2^n - 1) / 2).
    # For n=12, this is ceil(4095/2) = 2048.
    # The set of powers of 2, {1, 2, 4, ..., 2^(n-1)}, achieves this bound.
    # Thus, the largest element is 2^(n-1).
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
