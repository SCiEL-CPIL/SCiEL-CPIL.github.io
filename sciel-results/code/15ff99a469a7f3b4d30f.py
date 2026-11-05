"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Directly constructs the optimal 100-bit binary string for OneMax.
    The optimal solution is a string of length N consisting entirely of '1's.
    This is the global optimum, so no search or evolution is required.
    """
    # The maximum number of ones in a binary string of length N is N.
    # Achieved by setting all bits to '1'.
    return "1" * N
# EVOLVE-BLOCK-END
