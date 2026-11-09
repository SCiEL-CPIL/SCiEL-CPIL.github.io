"""Spectral partitioning heuristic for ACO CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using spectral clustering of the distance graph.
    Edges within dense spectral clusters get high attractiveness; edges crossing
    spectral cut boundaries get suppressed.
    """
    n = distance_matrix.shape[0]
    t0 = time.time()
    DEADLINE = 250.0  # safety margin before 300s kill

    # Build a weighted graph: weight = 1/distance (higher weight = closer)
    # Use a k-nearest neighbor sparsification to make the graph tractable
    k = max(5, min(15, n // 3))
    
    # Compute inverse distance weights (closer = higher weight)
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    np.fill_diagonal(inv_dist, 0.0)
    
    # KNN sparsification: keep only k nearest neighbors per node
    knn_mask = np.zeros_like(inv_dist, dtype=bool)
    for i in range(n):
        row = inv_dist[i]
        # Get indices of k largest values (excluding self which is 0)
        idx = np.argpartition(row, -k)[-k:]
        knn_mask[i, idx] = True
    # Symmetrize
    knn_mask = knn_mask | knn_mask.T
    
    # Sparse weighted adjacency
    W = inv_dist * knn_mask
    np.fill_diagonal(W, 0.0)
    
    # Degree matrix
    degree = W.sum(axis=1)
    # Avoid division by zero
    degree_safe = np.where(degree > 0, degree, 1.0)
    
    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Symmetrize to avoid numerical issues
    L_sym = (L_sym + L_sym.T) / 2.0
    
    # Compute Fiedler vector (second smallest eigenvalue's eigenvector)
    # For n=51, full eigendecomposition is fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    except np.linalg.LinAlgError:
        # Fallback: simple inverse distance heuristic
        return 1.0 / (distance_matrix + 1e-9)
    
    # Sort by eigenvalue
    idx_sort = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx_sort]
    
    # Fiedler vector is the second eigenvector (index 1)
    fiedler = eigenvectors[:, 1]
    
    # Also get the third eigenvector if available for 2D embedding
    if n > 2:
        fiedler2 = eigenvectors[:, 2]
    else:
        fiedler2 = np.zeros(n)
    
    # Compute spectral distance: edges within same spectral cluster should have
    # similar Fiedler values. We use the difference in Fiedler values as a
    # "crossing" penalty.
    fiedler_diff = np.abs(fiedler[:, None] - fiedler[None, :])
    if n > 2:
        fiedler2_diff = np.abs(fiedler2[:, None] - fiedler2[None, :])
    else:
        fiedler2_diff = np.zeros_like(fiedler_diff)
    
    # Combined spectral distance
    spectral_dist = fiedler_diff + 0.5 * fiedler2_diff
    
    # Normalize spectral distance to [0, 1]
    spectral_max = spectral_dist.max()
    if spectral_max > 0:
        spectral_dist_norm = spectral_dist / spectral_max
    else:
        spectral_dist_norm = spectral_dist
    
    # Heuristic: high for close nodes that are in the same spectral cluster
    # base attractiveness: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    np.fill_diagonal(base, 0.0)
    
    # Spectral boost: edges with small spectral distance get boosted
    # Edges with large spectral distance get penalized
    spectral_factor = 1.0 + 2.0 * (1.0 - spectral_dist_norm)
    
    # Combine: base distance heuristic * spectral factor
    heuristic = base * spectral_factor
    
    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all non-diagonal entries are positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        # Return a safe fallback
        return 1.0 / (distance_matrix + 1e-9)
    
    # Validate
    if not np.all(np.isfinite(heuristic)):
        return 1.0 / (distance_matrix + 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
