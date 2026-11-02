"""Optimized coefficients for the uncertainty inequality bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # Prior state of the art (Goncalves et al. 2017) as listed in the notebook.
    # These coefficients minimize the upper bound on C4 for the Hermite-Gaussian
    # test function with H0, H4, H8 terms (H12 fixed to satisfy g(0)=0).
    return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END
