"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones ("1" * 100), which gives a score of 100.
    We return this directly since it is the provably optimal solution.
    """
    return "1" * N
# EVOLVE-BLOCK-END
