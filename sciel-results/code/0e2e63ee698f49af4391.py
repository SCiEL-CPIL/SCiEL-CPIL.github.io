"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The objective is OneMax: maximize the number of 1s in the string.
    The global optimum is a string of all 1s, which has 100 ones.
    This is a deterministic construction that returns the known optimum.
    """
    return "1" * N
# EVOLVE-BLOCK-END
