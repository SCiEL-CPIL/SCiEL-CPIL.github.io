"""OneMax direct binary construction (100 bits) - maximize ones."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones.
    
    For OneMax, the optimal solution is all ones.
    We return the string of N ones directly.
    """
    return "1" * N
# EVOLVE-BLOCK-END
