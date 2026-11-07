"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse-squared distance as the heuristic, which provides
    stronger guidance toward short edges than the seed's inverse distance.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    # Replace diagonal (self-loops) with a large value to discourage them
    n = d.shape[0]
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    # Inverse squared distance: sharper contrast between short and long edges
    eta = 1.0 / (d_safe ** 2 + 1e-18)
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
