"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly N/2.
    The theoretical minimum for the maximum overlap with the shifted complement
    is N/4, achieved by the uniform distribution h=0.5.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
