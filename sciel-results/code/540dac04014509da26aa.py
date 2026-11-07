"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        return np.ones_like(d, dtype=np.float64)
    
    # Set diagonal to a large value so 1/d^alpha becomes very small
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Use inverse cubic distance for sharper guidance toward short edges
    alpha = 3.0
    eta = 1.0 / (d_safe ** alpha + 1e-30)
    
    # Compute MST to identify edges that are likely part of good tours
    # MST edges get a boost to encourage connectivity
    try:
        mst = minimum_spanning_tree(d_safe)
        mst_dense = mst.toarray()
        # MST edges get a multiplicative boost
        mst_boost = 1.0 + 0.5 * (mst_dense > 0)
        eta = eta * mst_boost
    except Exception:
        pass
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
