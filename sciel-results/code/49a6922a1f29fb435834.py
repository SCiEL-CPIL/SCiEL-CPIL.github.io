"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
DEADLINE_SECONDS = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for OneMax.
    Directly constructs the all-ones string, which is the global optimum.
    """
    # The global optimum for OneMax is a string of N ones.
    # This is the most direct and efficient construction.
    return '1' * N
# EVOLVE-BLOCK-END
