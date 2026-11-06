"""
Rewritten program for OneMax (100 bits).
Uses a pre-computed optimal solution for zero-overhead construction.
"""

N = 100

# Pre-compute the optimal string at module load time.
# This ensures construct() returns in constant time with no branching or computation.
_OPTIMAL_STRING = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal 100-bit binary string maximizing the number of ones.
    The optimal solution for OneMax is a string of all ones.
    """
    return _OPTIMAL_STRING
# EVOLVE-BLOCK-END
