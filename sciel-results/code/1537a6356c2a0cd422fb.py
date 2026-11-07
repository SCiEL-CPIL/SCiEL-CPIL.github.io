# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using an additive repulsive force model.
    
    We add a penalty to the distance of each edge proportional to its usage count
    and inversely proportional to its length. This creates a strong repulsive force
    for short, frequently used edges, forcing the local search to explore alternative
    connections.
    """
    n = edge_distance.shape[0]
    
    if edge_distance is None or edge_distance.shape[0] != edge_distance.shape[1]:
        return edge_distance.copy()
    
    # Copy the true distance matrix
    penalized = edge_distance.copy()
    
    # Ensure diagonal is 0
    np.fill_diagonal(penalized, 0.0)
    
    # Calculate inverse distance weights
    # Add epsilon to avoid division by zero
    eps = 1e-8
    inv_dist = 1.0 / (edge_distance + eps)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Normalize inverse distances to [0, 1] to control the magnitude of the penalty
    # This ensures the "shortest" edges get a weight of 1.0, and longer ones get less
    inv_dist_max = inv_dist.max()
    if inv_dist_max > 0:
        inv_dist_norm = inv_dist / inv_dist_max
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Usage counts
    usage = edge_n_used.astype(float)
    
    # The penalty is additive.
    # We want the penalty to be significant but not overwhelming.
    # The average edge length in a unit square TSP with 100 points is ~0.05-0.1.
    # Max usage is 20.
    # If we use alpha = 0.05, max penalty is 0.05 * 20 * 1.0 = 1.0.
    # This is large compared to edge length, which might be good for escaping deep local minima.
    # Let's start with a moderate alpha.
    
    alpha = 0.05
    
    # Apply penalty: d_new = d_old + alpha * usage * inv_dist_norm
    # Only apply to off-diagonal elements where usage > 0? 
    # Adding 0 is fine, but it's cleaner to just add.
    
    penalty = alpha * usage * inv_dist_norm
    
    # Ensure symmetry (just in case, though inputs should be symmetric)
    penalty = 0.5 * (penalty + penalty.T)
    
    penalized = penalized + penalty
    
    # Ensure diagonal is 0
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, 0.0)
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
# EVOLVE-BLOCK-END
