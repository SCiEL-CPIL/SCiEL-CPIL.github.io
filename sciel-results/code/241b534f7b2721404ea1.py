"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of 31 distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    The set is the arithmetic progression {226, 227, ..., 256}.
    """
    return list(range(226, 257))

# EVOLVE-BLOCK-END
