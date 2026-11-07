"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Set diagonal to large value
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Compute MST-based heuristic
    # MST edges get high attractiveness to enforce connectivity
    # Non-MST edges get score inversely proportional to distance
    
    # Base heuristic: unnormalized squared inverse distance
    eta = 1.0 / (d_safe ** 2)
    
    # Compute MST to identify important edges for connectivity
    # Convert to sparse graph for MST computation
    from scipy.sparse import csr_matrix
    graph = csr_matrix(d_safe)
    mst = minimum_spanning_tree(graph)
    mst_dense = mst.toarray()
    
    # Boost MST edges multiplicatively
    # MST edges get a boost factor to make them significantly more attractive
    mst_mask = (mst_dense > 0)
    boost_factor = 10.0
    eta[mst_mask] *= boost_factor
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
