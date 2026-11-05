"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of n=12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    Uses the property that the set of powers of 2 {1, 2, 4, ..., 2^(n-1)}
    provides distinct subset sums and minimizes the largest element.
    """
    n = 12
    # Generate powers of 2 from 2^0 to 2^(n-1)
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
