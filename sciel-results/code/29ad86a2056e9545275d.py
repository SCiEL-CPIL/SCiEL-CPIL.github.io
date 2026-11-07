"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the TSP heuristic.
    
    Strategy:
    1. Compute inverse distance (base attractiveness).
    2. Identify a sparse set of 'high-value' edges using a combination of:
       - Proximity to nearest neighbors (local structure)
       - Degree-based centrality (avoiding hub-like nodes in the heuristic)
    3. Use a non-linear transformation to emphasize short, high-quality edges
       while suppressing long, low-quality ones more aggressively than 1/d.
    4. Add a small noise-free boost for edges that are among the top-k shortest
       for either endpoint, to guide ants towards locally optimal structures.
    
    This structural approach aims to create a heuristic that strongly biases
    ants towards constructing tours with low local edge costs, which correlates
    with globally shorter tours in uniform random point instances.
    """
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    safe_dist = np.where(distance_matrix > 1e-9, distance_matrix, 1.0)
    base = 1.0 / safe_dist
    
    # Identify nearest neighbors for each node
    # For each row, find the indices of the k shortest distances
    k = max(3, n // 10)  # Adaptive k
    k = min(k, n - 1)    # Can't be more than n-1
    
    # Get the k smallest values per row
    # argsort is O(n log n) per row, total O(n^2 log n)
    # For n=50, this is very fast
    
    # Create a mask of "good" edges: edges that are among the k shortest for i or j
    good_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        # Sort distances from node i
        sorted_idx = np.argsort(distance_matrix[i])
        # Take the k closest (excluding self)
        # Note: distance_matrix[i][i] is 0, so it will be first
        # We want the k closest OTHER nodes
        nn_indices = sorted_idx[1:k+1]
        good_mask[i, nn_indices] = True
        # Symmetry: if i->j is good, j->i should also be considered
        good_mask[nn_indices, i] = True
    
    # Apply a boost to good edges
    # Use a power law to make the heuristic more selective
    # Higher power = more selective (emphasizes very short edges)
    # We want to make good edges significantly more attractive
    
    # Transform: h = (base * boost_factor) for good edges, base for others
    # But we need to be careful not to create too much disparity
    
    # Alternative: Use a non-linear scaling of the base heuristic
    # h = base^alpha, where alpha > 1 makes short edges relatively more attractive
    
    alpha = 2.0  # Power exponent
    
    # Apply power transformation to base
    transformed = base ** alpha
    
    # Boost good edges further
    # Multiply good edges by an additional factor
    boost_factor = 1.5
    
    result = transformed.copy()
    result[good_mask] *= boost_factor
    
    # Ensure all values are positive
    # The task says "Heuristic entries that are not positive become 1e-9"
    # But we should make sure our output is positive
    
    # Normalize to avoid extreme values? The task doesn't require normalization,
    # but very large values might cause numerical issues in probability computation
    # Let's keep it simple and just ensure positivity
    
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
