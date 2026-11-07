"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a combination of:
    1. Base inverse distance with tuned exponent
    2. Smooth distance-aware NN boost (continuous, not binary)
    3. Mutual NN detection for extra boost
    """
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    dm = distance_matrix.astype(np.float64).copy()
    diag_mask = np.eye(n, dtype=bool)
    dm_safe = dm.copy()
    dm_safe[diag_mask] = 1.0
    
    # Base heuristic: inverse distance with exponent ~2 for stronger short-edge preference
    base = 1.0 / (dm_safe + 1e-9) ** 2.0
    base[diag_mask] = 0.0
    
    # Find k-nearest neighbors for each node
    k = min(5, n - 1)
    nn_edges = np.zeros((n, n), dtype=bool)
    nn_rank = np.zeros((n, n), dtype=np.float64)  # rank-based smooth weight
    for i in range(n):
        row = dm_safe[i].copy()
        row[i] = np.inf
        idx = np.argpartition(row, k)[:k]
        nn_edges[i, idx] = True
        # Compute smooth rank weights: rank 0 (nearest) gets highest weight
        sorted_idx = idx[np.argsort(row[idx])]
        for rank, j in enumerate(sorted_idx):
            # Smooth decay: weight = 1 / (1 + rank * 0.5)
            nn_rank[i, j] = 1.0 / (1.0 + rank * 0.5)
    
    # Identify mutual nearest neighbors (using k=1 for mutual)
    mutual = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = dm_safe[i].copy()
        row[i] = np.inf
        nearest = np.argmin(row)
        if nn_edges[nearest, i]:
            mutual[i, nearest] = True
            mutual[nearest, i] = True
    
    # Apply smooth boost: base * (1 + boost_factor * nn_rank)
    boost_factor = 8.0
    boosted = base * (1.0 + boost_factor * nn_rank)
    
    # Extra boost for mutual NN
    boosted[mutual] *= 3.0
    
    # Ensure diagonal is zero
    boosted[diag_mask] = 0.0
    
    # Clip to ensure positivity
    boosted = np.where(boosted > 0, boosted, 1e-9)
    
    # Don't normalize to max=1; keep natural scale
    # The pheromone starts at 1, so heuristic scale relative to 1 matters
    # Scale so median is around 1 for balanced interaction with pheromone
    med = np.median(boosted[~diag_mask])
    if med > 0:
        boosted = boosted / med
    
    return boosted
# EVOLVE-BLOCK-END
