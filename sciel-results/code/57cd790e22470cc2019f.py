"""OneMax direct binary construction (100 bits) - all ones is optimal."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution is all ones: '1' * 100.
    This is a deterministic, instantaneous solution.
    """
    # The OneMax objective is simply the count of '1' bits.
    # The maximum possible value is N (all ones).
    # Return the string of all ones.
    return "1" * N
# EVOLVE-BLOCK-END
