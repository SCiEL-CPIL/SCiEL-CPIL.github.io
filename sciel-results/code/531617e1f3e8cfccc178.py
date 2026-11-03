"""Rewritten program for the suite task; uses the theoretical optimum constant sequence."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal constant sequence.
    By Cauchy-Schwarz, (sum h)^2 <= N * sum h^2 = N * C(0).
    Thus max|C(k)| >= C(0) >= (sum h)^2 / N.
    The ratio is >= 1/N.
    The constant sequence h_i=1 achieves ratio 1/N exactly.
    """
    return np.ones(N)

# EVOLVE-BLOCK-END
