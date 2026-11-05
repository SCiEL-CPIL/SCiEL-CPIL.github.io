"""Optimal program for the suite task; achieves the theoretical minimum value of 2.0."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # The value is bounded below by 2.0 by Cauchy-Schwarz:
    # max(k) >= k[0] = sum(h^2)
    # V >= 2*N*sum(h^2)/(sum(h))^2 >= 2
    # Equality holds for constant vectors.
    # We construct a constant vector using np.full.
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
