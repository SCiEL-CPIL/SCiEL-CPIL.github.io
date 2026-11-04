"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    """Return 50 nonnegative step heights that maximize the ratio.
    
    The optimal solution is a single spike: all mass at one position.
    This achieves ratio = 1.0 (the theoretical maximum), since:
    - ||h||_2^2 = 1, ||h||_1 = 1, ||h*h||_inf = 1
    - Ratio = 1 / (1 * 1) = 1
    
    The seed (all ones) gives ratio = 50/(50*50) = 0.02, so this is 50x better.
    """
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
