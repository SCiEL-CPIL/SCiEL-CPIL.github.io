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
    Returns the global optimum directly to minimize overhead.
    """
    # The global optimum for OneMax is a string of all '1's.
    # Direct construction avoids any loop or join overhead.
    return '1' * N
# EVOLVE-BLOCK-END
