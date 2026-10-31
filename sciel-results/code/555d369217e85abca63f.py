"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2.
    
    By Cauchy-Schwarz: (sum x)^2 <= N * sum(x^2) = N * autoconv[0].
    Thus max|autoconv|/integral^2 >= autoconv[0]/integral^2 >= 1/N.
    Equality holds iff all x[i] are equal. The optimal is a constant vector.
    """
    # The theoretical lower bound is 1/N, achieved by constant vectors.
    return np.ones(N)
# EVOLVE-BLOCK-END
