"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The OneMax objective function is to maximize the count of '1' bits.
    The global optimum is a string of all '1's.
    """
    return "1" * N
# EVOLVE-BLOCK-END
