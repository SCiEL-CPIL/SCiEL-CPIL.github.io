"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Theoretical analysis shows that the constant sequence minimizes
    # max_k |c_k| / (sum x_i)^2.
    # For x_i = 1, c_k = N - |k|, max |c_k| = N, sum x_i = N.
    # Score = N / N^2 = 1/N.
    # Any deviation from constant increases sum(x^2) relative to (sum x)^2.
    # We construct the optimal constant sequence.
    x = np.ones(N)
    return x
# EVOLVE-BLOCK-END
