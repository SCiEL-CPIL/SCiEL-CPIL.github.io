"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    d = dm + 1e-9
    h = 1.0 / d
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    h = np.maximum(h, 1e-9)
    return h
# EVOLVE-BLOCK-END
