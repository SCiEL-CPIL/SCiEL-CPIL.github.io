"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with penalties applied to frequently used edges.
    """
    n = edge_distance.shape[0]
    
    # Normalize edge usage counts to [0, 1]
    max_used = edge_n_used.max()
    if max_used > 0:
        normalized_usage = edge_n_used / max_used
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Identify edges in the current local optimum tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
    
    # Penalty strategy:
    # 1. Base penalty proportional to usage frequency (standard GLS)
    # 2. Additional penalty for edges in the current local optimum to encourage escape
    # 3. Use a non-linear scaling to make penalties more aggressive for heavily used edges
    
    # Non-linear usage scaling: emphasize high-usage edges more
    usage_penalty = 0.15 * (normalized_usage ** 1.5)
    
    # Tour-specific penalty: penalize edges in the current local optimum
    # This helps break out of the current local solution
    tour_penalty = np.zeros((n, n), dtype=float)
    tour_penalty[tour_edges] = 0.08
    
    # Combine penalties
    total_penalty = usage_penalty + tour_penalty
    
    # Apply penalty multiplicatively to avoid zeroing out small distances
    # Use a soft cap to prevent excessive distortion
    penalty_multiplier = 1.0 + total_penalty
    
    # Cap the multiplier to avoid extreme values
    penalty_multiplier = np.clip(penalty_multiplier, 1.0, 3.0)
    
    # Create perturbed distance matrix
    perturbed = edge_distance * penalty_multiplier
    
    # Ensure diagonal remains zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure symmetry (in case of any floating point issues)
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure all values are finite
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
