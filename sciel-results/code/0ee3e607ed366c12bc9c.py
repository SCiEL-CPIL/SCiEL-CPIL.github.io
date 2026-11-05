N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones.
    """
    # The global optimum for OneMax is all bits set to 1.
    # We construct this directly.
    return "1" * N
# EVOLVE-BLOCK-END
