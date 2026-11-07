"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix for ACO TSP.
    Uses a neighborhood-restricted inverse distance heuristic.
    Only edges to the k-nearest neighbors of each node have significant heuristic values.
    """
    n = distance_matrix.shape[0]
    k = 10  # Number of nearest neighbors to consider
    
    # Initialize heuristic with very small values (will be clipped to 1e-9 by evaluator if 0)
    heuristic = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute k-nearest neighbors for each node
    # argsort gives indices that would sort the array
    # We take the first k indices for each row
    # distance_matrix has 0 on diagonal, so node i is its own 0th neighbor.
    # We should probably exclude self or just let it be, but ants can't visit visited nodes.
    # Since diagonal is 0, 1/0 is inf. We handle this by adding epsilon or masking.
    
    # Get indices of k smallest distances for each row
    # Note: distance_matrix[i][i] is 0.
    # We want j != i.
    # Let's mask the diagonal with a large value before sorting to ensure self is not picked if k < n.
    # But since k=10 and n=50, self (0th) will be picked.
    # We can just set diagonal to inf before sorting, or ignore index 0 in the selection if it's self.
    
    d = distance_matrix.copy()
    # Set diagonal to infinity so it's not selected as a nearest neighbor
    np.fill_diagonal(d, np.inf)
    
    # Get k nearest neighbor indices for each node
    # argsort along axis 1
    # shape (n, n)
    # We only need the first k columns
    nn_indices = np.argpartition(d, k, axis=1)[:, :k]
    
    # Create a mask for the neighborhood
    # mask[i, j] is True if j is in nn_indices[i]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)
    # Use advanced indexing to set mask to True for the selected neighbors
    mask[rows, nn_indices] = True
    
    # Calculate inverse distance
    # Add epsilon to avoid division by zero (though diagonal is inf, off-diagonal distances are > 0)
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Apply mask: only keep heuristic values for neighbors
    # Non-neighbors get 0.0, which the evaluator will convert to 1e-9
    heuristic = np.where(mask, inv_dist, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
