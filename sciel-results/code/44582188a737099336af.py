"""Guided local search penalty rule for TSP on 3 fixed 100-point instances."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    
    Returns a modified distance matrix that penalizes frequently-used edges
    and edges near the current local optimum, encouraging 2-opt to explore
    structurally different tours.
    """
    n = edge_distance.shape[0]
    
    # Get the set of edges in the current tour
    tour_edges = set()
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(i, j), max(i, j)))
    
    # Create a boolean mask for tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    for (i, j) in tour_edges:
        tour_mask[i, j] = True
        tour_mask[j, i] = True
    
    # Scale factor: median of off-diagonal distances
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_dists = edge_distance[off_diag_mask]
    dist_scale = np.median(off_diag_dists) if len(off_diag_dists) > 0 else 1.0
    
    # Base penalty for edges in the current tour, scaled by usage
    # Use a stronger penalty for heavily-used edges
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        # Normalize usage to [0, 1]
        normalized_usage = edge_n_used / (1.0 + max_used)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Penalty for tour edges: proportional to usage, with a base penalty
    # The base ensures even first-time tour edges get penalized
    base_penalty = 0.5 * dist_scale
    usage_penalty = 1.0 * dist_scale * normalized_usage
    
    # Build the modified distance matrix
    modified = edge_distance.copy()
    
    # Apply penalty to tour edges
    modified[tour_mask] += base_penalty + usage_penalty[tour_mask]
    
    # Repulsive force for edges near the tour (exclusion zone)
    # For each edge (i,j) not in the tour, penalize based on how close
    # it is to tour edges in metric space
    # Use a cheap approximation: penalize edges that share a vertex with
    # a tour edge, with strength based on usage of the shared tour edge
    
    # Precompute for each vertex, the max usage of incident tour edges
    vertex_tour_usage = np.zeros(n)
    for (i, j) in tour_edges:
        usage_ij = edge_n_used[i, j]
        vertex_tour_usage[i] = max(vertex_tour_usage[i], usage_ij)
        vertex_tour_usage[j] = max(vertex_tour_usage[j], usage_ij)
    
    # For non-tour edges, add a small penalty if they share a vertex with a tour edge
    # This creates a soft exclusion zone
    non_tour_mask = ~tour_mask & off_diag_mask
    # For each non-tour edge (i,j), penalty based on max of vertex_tour_usage[i], vertex_tour_usage[j]
    # Vectorized: create a matrix of max usage at either endpoint
    max_vertex_usage = np.maximum(vertex_tour_usage[:, np.newaxis], 
                                   vertex_tour_usage[np.newaxis, :])
    
    # Only apply to non-tour, off-diagonal edges
    repulsion_strength = 0.15 * dist_scale
    repulsion = np.zeros((n, n))
    repulsion[non_tour_mask] = repulsion_strength * max_vertex_usage[non_tour_mask] / (1.0 + max_used)
    
    modified += repulsion
    
    # Ensure diagonal is zero
    np.fill_diagonal(modified, 0.0)
    
    # Ensure symmetry
    modified = 0.5 * (modified + modified.T)
    
    # Ensure all values are finite and non-negative
    modified = np.where(np.isfinite(modified) & (modified >= 0), modified, 0.0)
    
    return modified
# EVOLVE-BLOCK-END
