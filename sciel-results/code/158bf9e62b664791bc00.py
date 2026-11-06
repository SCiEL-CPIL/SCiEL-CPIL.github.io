"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    """
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = 10 * 12 = 120.
    return 143
# EVOLVE-BLOCK-END
