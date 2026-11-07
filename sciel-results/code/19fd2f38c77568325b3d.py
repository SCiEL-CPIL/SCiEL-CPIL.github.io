"""Heuristic matrix for ACO TSP using MST backbone + inverse distance blend."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _mst_edges(n, dist):
    """Return set of (i,j) edges in the MST using Prim's algorithm."""
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    edges = []
    parent = np.full(n, -1, dtype=int)
    for _ in range(n):
        # Find the unvisited node with minimum distance to the tree
        unvisited = ~in_mst
        if not unvisited.any():
            break
        # Use argmin over the masked array
        masked = np.where(unvisited, min_dist, np.inf)
        u = int(np.argmin(masked))
        if min_dist[u] == np.inf:
            break
        in_mst[u] = True
        if parent[u] != -1:
            i, j = u, parent[u]
            if i > j:
                i, j = j, i
            edges.append((i, j))
        # Update distances
        for v in range(n):
            if not in_mst[v] and dist[u, v] < min_dist[v]:
                min_dist[v] = dist[u, v]
                parent[v] = u
    return edges


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for ACO TSP.
    Combines inverse distance with MST structural guidance.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)

    # Ensure symmetric and handle diagonal
    dm = np.maximum(dm, dm.T)
    np.fill_diagonal(dm, 0.0)

    # Base heuristic: inverse distance (stronger for closer nodes)
    # Use a moderate exponent to avoid extreme values
    eps = 1e-9
    base = 1.0 / (dm + eps)

    # Normalize base to [0, 1] range for blending
    base_max = base.max()
    if base_max > 0:
        base_norm = base / base_max
    else:
        base_norm = np.ones_like(base)

    # Compute MST edges for structural guidance
    mst_edges = _mst_edges(n, dm)
    mst_mask = np.zeros((n, n), dtype=np.float64)
    for (i, j) in mst_edges:
        mst_mask[i, j] = 1.0
        mst_mask[j, i] = 1.0

    # Blend: boost MST edges significantly
    # MST edges get a high constant, others get the normalized inverse distance
    mst_boost = 1.0       # high attractiveness for MST edges
    non_mst_scale = 0.3   # lower attractiveness for non-MST edges

    heuristic = non_mst_scale * base_norm
    heuristic[mst_mask > 0.5] = mst_boost

    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)

    # Ensure all non-diagonal entries are positive
    # The task says non-positive entries become 1e-9
    mask_nonzero = heuristic < 1e-9
    mask_nonzero[~np.eye(n, dtype=bool)] = mask_nonzero[~np.eye(n, dtype=bool)]
    # Actually, let's just clamp small values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    # But diagonal should remain 0 or very small; the ACO won't pick visited nodes anyway
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
# EVOLVE-BLOCK-END
