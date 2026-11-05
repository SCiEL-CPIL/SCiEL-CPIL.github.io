"""Optimal program for the suite task; achieves the theoretical minimum value of 2.0."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # The value is bounded below by 2.0.
    # Equality holds if and only if h is a constant vector.
    # We return a constant vector of 1s.
    return np.ones(N)
# EVOLVE-BLOCK-END
