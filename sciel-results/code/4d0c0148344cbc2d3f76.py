"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a linear combination of:
    1. MST edge indicator: high weight to enforce connectivity
    2. Inverse distance: geometric locality for all edges
    """
    n = distance_matrix.shape[0]
    
    if n <= 1:
        return np.ones_like(distance_matrix, dtype=float)
    
    dist = np.array(distance_matrix, dtype=float, copy=True)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    
    # Replace non-finite/negative
    dist = np.where(np.isfinite(dist), dist, 1e6)
    dist = np.where(dist > 0, dist, 1e-9)
    
    # 1. Compute MST mask
    try:
        mst_graph = minimum_spanning_tree(dist, directed=False)
        mst_matrix = mst_graph.toarray()
        mst_mask = np.isfinite(mst_matrix) & (mst_matrix > 0)
        mst_mask = mst_mask | mst_mask.T
        np.fill_diagonal(mst_mask, False)
    except Exception:
        mst_mask = np.zeros_like(dist, dtype=bool)
    
    # 2. Base heuristic: inverse distance with power 2
    # Using power 2 instead of 3 for a smoother gradient
    p = 2.0
    base_heuristic = 1.0 / (dist + 1e-9) ** p
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 0.0)
    np.fill_diagonal(base_heuristic, 0.0)
    
    # 3. Linear combination:
    # MST edges get a strong additive boost
    # Non-MST edges rely on inverse distance
    # Weight for MST component
    w_mst = 10.0
    # Weight for inverse distance component
    w_dist = 1.0
    
    # Normalize base heuristic to be comparable with MST weight
    # Scale base heuristic so that median is around 1
    median_base = np.median(base_heuristic[base_heuristic > 0])
    if median_base > 0:
        base_heuristic = base_heuristic / median_base
    
    # Combine: heuristic = w_mst * mst_mask + w_dist * base_heuristic
    heuristic = w_mst * mst_mask.astype(float) + w_dist * base_heuristic
    
    # 4. Cleanup
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
