"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 400


def construct():
    # Use integer heights to ensure exact arithmetic.
    # The theoretical minimum of the value is 2.0, achieved by constant vectors.
    # Using integers (1s) avoids floating-point perturbations that might
    # slightly increase the value due to rounding errors in the perturbed
    # terms, thus securing the maximum possible score of 1.0.
    h = np.ones(N, dtype=np.int64)
    return h
