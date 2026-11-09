"""OneMax direct binary construction (100 bits)."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string with all bits set to 1.
    This is the global optimum for the OneMax problem, maximizing the number of ones.
    """
    return "1" * N
# EVOLVE-BLOCK-END
