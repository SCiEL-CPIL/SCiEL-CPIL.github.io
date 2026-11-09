"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Strategy: Use simple integer locations z_i = i for i=1..12.
    This creates a compact cluster near the origin which may reduce
    the largest positive root of the quotient polynomial q(x).
    """
    z = [float(i) for i in range(1, 13)]
    return z
# EVOLVE-BLOCK-END
