"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones, which gives a score of 100.
    Uses integer bit manipulation to generate the string.
    """
    return bin(2**N - 1)[2:]
# EVOLVE-BLOCK-END
