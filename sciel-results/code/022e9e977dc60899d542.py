"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones: "1" * 100.
    We return it directly. This is provably optimal for the OneMax objective.
    Score: 100 (maximum possible).
    """
    return "1" * N
# EVOLVE-BLOCK-END
