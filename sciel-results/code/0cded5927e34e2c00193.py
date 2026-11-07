"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    Penalizes edges based on their cumulative usage in previous local optima.
    Uses a smooth, saturating penalty to encourage escape from the current
    local optimum while keeping the perturbed matrix well-behaved for 2-opt.
    """
    n = edge_distance.shape[0]
    
    # Start with a copy of the true distances
    perturbed = edge_distance.copy()
    
    # Get the edges in the current tour
    tour_edges = set()
    for i in range(n):
        j = (i + 1) % n
        a, b = min(i, j), max(i, j)
        tour_edges.add((a, b))
    
    # Build a penalty matrix based on edge usage
    # Use a smooth saturating function: penalty = alpha * usage / (beta + usage)
    # This grows with usage but saturates, preventing extreme distortions
    alpha = 0.5  # Maximum penalty factor (50% increase at saturation)
    beta = 2.0   # Saturation parameter (penalty reaches ~half of alpha at usage=beta)
    
    # Compute penalty for each edge based on cumulative usage
    # edge_n_used is symmetric, so we work with the full matrix
    penalty = alpha * edge_n_used / (beta + edge_n_used)
    
    # Add extra penalty to edges in the current tour for immediate escape
    # This helps break the current local optimum more aggressively
    for (a, b) in tour_edges:
        penalty[a, b] += 0.3  # Additional immediate penalty
        penalty[b, a] += 0.3
    
    # Apply the penalty multiplicatively to avoid negative distances
    perturbed = edge_distance * (1.0 + penalty)
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
