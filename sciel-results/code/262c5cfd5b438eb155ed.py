"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    # Theoretical minimum is 1/N, achieved by constant sequence.
    # Return the optimal sequence directly.
    return np.ones(N)
# EVOLVE-BLOCK-END
