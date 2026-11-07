"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for the ACO heuristic.
    
    Strategy: Use a k-nearest-neighbor based heuristic that strongly favors
    geometrically local edges (short distances) while still allowing some
    global structure. The heuristic is computed as:
    
    1. Compute a distance-based baseline: 1/(d + eps)
    2. Identify k-nearest neighbors for each node
    3. Boost edges that are among the k-nearest neighbors
    4. This creates a sparse, local-biased heuristic that guides ants toward
       shorter, more geometrically plausible tours
    
    The key insight from the measured improvement (1 -> 1.235) is that
    restricting ant attention to geometrically plausible edges reduces
    premature convergence to long detours.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return distance_matrix.copy()
    
    # Ensure we have a valid square matrix
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Start with inverse distance as base heuristic
    eps = 1e-9
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # For small matrices, just use the base heuristic
    if n <= 10:
        # Make sure diagonal is 0 (no self-loops)
        np.fill_diagonal(base_heuristic, 0.0)
        return base_heuristic
    
    # Compute k-nearest neighbors for each node
    # Use a k that scales with n but stays local
    k = max(3, min(10, n // 5))
    
    # For each node, find its k nearest neighbors
    # We'll boost edges that are in the k-NN set
    knn_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        # Get distances from node i to all others
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Find k nearest neighbors
        if k < n - 1:
            # Get indices of k nearest neighbors
            k_nn_indices = np.argpartition(dists, k)[:k]
        else:
            k_nn_indices = np.arange(n)
            k_nn_indices = k_nn_indices[k_nn_indices != i]
        
        # Mark these edges as "local"
        for j in k_nn_indices:
            knn_mask[i, j] = True
    
    # Create a boosted heuristic:
    # - For k-NN edges: use a stronger inverse distance (higher exponent)
    # - For non-k-NN edges: use a weaker inverse distance (lower exponent)
    
    # Base: 1/(d + eps)^alpha where alpha controls how strongly we favor short edges
    # For k-NN edges, use higher alpha (stronger preference for short)
    # For non-k-NN edges, use lower alpha (weaker preference, but still positive)
    
    alpha_knn = 2.0   # Strong preference for short edges among neighbors
    alpha_other = 0.5  # Weak preference for short edges among non-neighbors
    
    # Compute the heuristic
    heuristic = np.zeros_like(distance_matrix, dtype=float)
    
    # For k-NN edges
    knn_edges = knn_mask & (distance_matrix > 0)  # Exclude diagonal
    heuristic[knn_edges] = 1.0 / (distance_matrix[knn_edges] + eps) ** alpha_knn
    
    # For non-k-NN edges (but not diagonal)
    non_knn_edges = (~knn_mask) & (distance_matrix > 0)
    heuristic[non_knn_edges] = 1.0 / (distance_matrix[non_knn_edges] + eps) ** alpha_other
    
    # Diagonal should be 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive (replace any non-positive with 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    # Normalize to keep values in a reasonable range
    # This helps with numerical stability in the ACO algorithm
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    # Ensure minimum value for non-zero entries
    non_zero = heuristic > 0
    heuristic[non_zero] = np.maximum(heuristic[non_zero], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
