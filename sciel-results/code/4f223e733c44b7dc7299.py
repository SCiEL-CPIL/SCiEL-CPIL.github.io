"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal set of n distinct positive integers with distinct subset sums
    # is the set of powers of 2: {1, 2, 4, ..., 2^(n-1)}.
    # The largest element is 2^(n-1).
    # For n=12, the largest element is 2^11 = 2048.
    # Any other set with distinct subset sums must have a largest element >= 2^(n-1).
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
