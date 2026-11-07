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
    
    try:
        # Compute MST using scipy
        mst = minimum_spanning_tree(d_safe)
        mst_dense = mst.toarray()
        
        # Identify MST edges (positive weights in MST matrix)
        mst_mask = (mst_dense > 0) & (mst_dense < 1e9)
        
        # Base heuristic: inverse distance (geometric locality)
        # Use alpha=2 for stronger preference for short edges
        base_eta = 1.0 / (d_safe ** 2)
        
        # Boost MST edges to enforce connectivity
        # MST edges get a multiplicative boost
        mst_boost = 5.0
        eta = base_eta.copy()
        eta[mst_mask] *= mst_boost
        
        # Also boost edges that are in the 2-MST (second-best spanning tree)
        # to provide alternative connectivity paths
        # Find edges that would be in a second MST by removing one MST edge at a time
        # For efficiency, just boost the next-best edges for each node
        
        # Additional boost for edges that connect nodes with high degree in MST
        # This helps maintain connectivity in the tour
        mst_degree = mst_mask.sum(axis=1)
        for i in range(n):
            for j in range(n):
                if i != j and not mst_mask[i, j]:
                    # Boost non-MST edges that connect high-degree MST nodes
                    if mst_degree[i] >= 2 and mst_degree[j] >= 2:
                        eta[i, j] *= 1.5
    except Exception:
        # Fallback: simple inverse distance heuristic
        eta = 1.0 / (d_safe ** 2)
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        # Return simple fallback
        eta_fallback = 1.0 / (d_safe ** 2)
        eta_fallback = np.where(np.isfinite(eta_fallback) & (eta_fallback > 0), eta_fallback, 1e-9)
        return eta_fallback
    
    return eta
# EVOLVE-BLOCK-END
