"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = dm.shape[0]

    # Work on a copy to avoid mutating input
    d = dm.copy()

    # Set diagonal to a large value so it gets low attractiveness
    np.fill_diagonal(d, np.inf)

    # Ensure no negative distances (shouldn't happen, but safe)
    d = np.maximum(d, 1e-9)

    # --- Topological (Delaunay-inspired) adjacency heuristic ---
    # For 50 points in the unit square, the Delaunay triangulation has ~150 edges.
    # We approximate a topological bias by identifying "local" neighbors.
    # For each node, we consider its k-nearest neighbors (k=6) as part of a local graph.
    # Edges in this local graph get a boost.
    # This mimics the sparsity and non-crossing property of Delaunay to some extent
    # without needing scipy.spatial (which is allowed but let's keep it simple and fast).
    # Actually, scipy is available. Let's use it if possible, but the task says
    # "Python standard library, NumPy and SciPy are available".
    # However, the current program is pure numpy. Let's stick to a numpy-based
    # k-NN approach which is robust and fast for n=50.

    k = 6  # Number of nearest neighbors to consider for the "topological" boost
    # For each node i, find the k nearest neighbors j.
    # Create a binary mask of these edges.
    # Since the graph is undirected, if i is a k-NN of j, we mark (i,j) and (j,i).
    # But strictly, k-NN is not symmetric. We can take the union.
    # Let's create a mask of size n x n, initialized to 0.
    local_mask = np.zeros((n, n), dtype=bool)

    # For each node, sort distances and pick top k
    # d[i, :] is the distance from i to all others
    # We need to ignore the diagonal (which is inf)
    for i in range(n):
        # Get indices of the k smallest distances
        # argsort returns indices that would sort the array
        # We want the first k indices
        # Note: d[i, i] is inf, so it will be at the end
        sorted_idx = np.argsort(d[i, :])
        # Take the first k
        neighbors = sorted_idx[:k]
        local_mask[i, neighbors] = True

    # Symmetrize the mask: if i is a neighbor of j, treat j as neighbor of i too
    # This creates a symmetric graph of "local" edges
    local_mask = local_mask | local_mask.T

    # --- Combine distance heuristic with topological boost ---
    # Base heuristic: inverse distance squared
    # h_base = 1 / d^2
    # We want to boost edges in the local graph.
    # Let's multiply by a factor.
    # If edge is local, multiply by 2.0 (or some boost factor)
    # If edge is not local, multiply by 1.0
    # But we also want to penalize non-local edges more strongly?
    # The idea is to prune the search space.
    # Let's try:
    # h = (1 / d^beta) * (boost if local else 1)
    
    beta = 2.0
    # d has inf on diagonal. 1/inf = 0.
    # d has 1e-9 on off-diagonal if distance was 0 (unlikely).
    # Let's compute h_base
    # Use np.power for efficiency
    h_base = 1.0 / np.power(d, beta)
    
    # Apply topological boost
    # Boost factor for local edges
    boost_factor = 2.0
    # Non-local edges get a penalty? Or just 1.0?
    # If we just boost local, the relative difference between local and non-local
    # is boost_factor.
    # Let's also consider that non-local edges are likely longer, so 1/d^2 already
    # penalizes them. The boost reinforces the short, local edges.
    
    h = h_base.copy()
    h[local_mask] *= boost_factor

    # Normalize to mean 1.0 over off-diagonal elements
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    valid_h = h[mask]
    if len(valid_h) > 0:
        mean_h = np.mean(valid_h)
        if mean_h > 0 and np.isfinite(mean_h):
            h = h / mean_h
        else:
            h = np.ones_like(dm, dtype=float)
    else:
        h = np.ones_like(dm, dtype=float)

    # Ensure non-finite values are handled
    h = np.where(np.isfinite(h), h, 1e-9)
    
    # Entries that are not positive become 1e-9
    h = np.where(h > 0, h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
