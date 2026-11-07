"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a combination of:
    1. Minimum Spanning Tree (MST) edges: high attractiveness to enforce connectivity
    2. Next Nearest Neighbor (NNN) edges: moderate boost to avoid local minima
    3. Non-special edges: inversely proportional to distance (power 3) for strong locality
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
    
    # 1. MST Mask
    try:
        mst_graph = minimum_spanning_tree(dist, directed=False)
        mst_matrix = mst_graph.toarray()
        mst_mask = np.isfinite(mst_matrix) & (mst_matrix > 0)
        mst_mask = mst_mask | mst_mask.T
        np.fill_diagonal(mst_mask, False)
    except Exception:
        mst_mask = np.zeros_like(dist, dtype=bool)
        
    # 2. Next Nearest Neighbor (NNN) Mask
    # argsort gives indices of sorted distances.
    # index 0 is self (dist 0), index 1 is 1st NN, index 2 is 2nd NN.
    try:
        sorted_indices = np.argsort(dist, axis=1)
        nnn_indices = sorted_indices[:, 2]
        nnn_mask = np.zeros_like(dist, dtype=bool)
        rows = np.arange(n)
        nnn_mask[rows, nnn_indices] = True
        # Symmetrize? NNN is not necessarily symmetric (i's 2nd NN is j, but j's 2nd NN might not be i).
        # However, for attractiveness, if i likes j, it's good. If j likes i, it's good.
        # We can just use the directed mask or symmetrize. Symmetrizing is safer for undirected TSP.
        nnn_mask = nnn_mask | nnn_mask.T
        np.fill_diagonal(nnn_mask, False)
    except Exception:
        nnn_mask = np.zeros_like(dist, dtype=bool)
        
    # 3. Base Heuristic
    # Power 3.0 for stronger locality than 2.0
    p = 3.0
    base_heuristic = 1.0 / (dist + 1e-9) ** p
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 0.0)
    np.fill_diagonal(base_heuristic, 0.0)
    
    # 4. Combine Boosts
    # MST boost: 4.0 (total factor 5.0)
    # NNN boost: 1.5 (total factor 2.5)
    # They can overlap. If an edge is both MST and NNN, it gets (1+4)*(1+1.5) = 12.5x base.
    mst_multiplier = 1.0 + 4.0 * mst_mask.astype(float)
    nnn_multiplier = 1.0 + 1.5 * nnn_mask.astype(float)
    
    heuristic = base_heuristic * mst_multiplier * nnn_multiplier
    
    # 5. Cleanup
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
