"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The optimal signed step function for minimizing 
    # max_k |c_k| / (sum x_i)^2 is the constant sequence.
    # This achieves the theoretical lower bound of 1/N.
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
