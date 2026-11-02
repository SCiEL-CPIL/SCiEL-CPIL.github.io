"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed equal-width step function that minimizes the maximum
    absolute autoconvolution (peak sidelobe level).
    
    Strategy: Randomized Search.
    The constant sequence has maximal sidelobes.
    Random sequences have sidelobes scaling with sqrt(N log N), which is
    significantly lower than N for large N.
    We generate a random sequence of +1/-1 with a fixed seed.
    """
    np.random.seed(42)
    return np.random.choice([-1, 1], size=N)
# EVOLVE-BLOCK-END
