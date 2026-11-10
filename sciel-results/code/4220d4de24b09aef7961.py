"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    # Set diagonal to a large value so it's effectively unattractive
    np.fill_diagonal(d, np.inf)
    # Sharpen the heuristic with a power of 2
    eps = 1e-9
    h = 1.0 / (d + eps) ** 2
    # Replace any non-finite values with a small positive number
    h[~np.isfinite(h)] = 1e-9
    # Row-normalize so each row sums to 1
    row_sums = h.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    h = h / row_sums
    # Ensure all values are positive
    h[h <= 0] = 1e-9
    # Re-normalize after clamping
    row_sums = h.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    h = h / row_sums
    return h
# EVOLVE-BLOCK-END
