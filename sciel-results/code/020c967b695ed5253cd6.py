"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses an iterative construction approach to build the optimal all-ones string.
    """
    # Build the string by iterating N times and appending '1'
    # This is functionally equivalent to "1" * N but structurally different.
    return "".join(['1'] * N)
# EVOLVE-BLOCK-END
