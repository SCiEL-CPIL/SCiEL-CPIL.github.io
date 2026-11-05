"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a sequence of 400 signed step heights.
    The theoretical minimum for the value is 2.0, achieved by any constant vector.
    np.ones(N) is the simplest such vector.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
