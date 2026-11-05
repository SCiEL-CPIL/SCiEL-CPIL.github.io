"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The theoretical lower bound for the objective function is 2.0,
    # achieved by any constant vector with non-zero sum.
    # The score is seed_value / value = 2.0 / value.
    # To maximize score, we must minimize value to 2.0.
    # Using np.full ensures a constant vector, which is optimal.
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
