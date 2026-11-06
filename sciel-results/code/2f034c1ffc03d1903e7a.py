"""
Optimized program for OneMax task.
Constructs a 100-bit binary string maximizing the number of ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    This is the direct, optimal solution for the OneMax problem.
    """
    # The optimal solution for OneMax is a string of all ones.
    # This is computed in O(N) time, which is negligible for N=100.
    # No search or iterative refinement is needed as the global optimum
    # is known analytically.
    return "1" * N
# EVOLVE-BLOCK-END
