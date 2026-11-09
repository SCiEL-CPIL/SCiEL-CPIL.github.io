"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    # Calculate the average edge length of the current tour to scale the penalty
    # This adapts the penalty strength to the specific instance geometry
    tour_edges = local_opt_tour[:-1]
    tour_next = local_opt_tour[1:]
    # Ensure we handle the closing edge if the tour is represented as a cycle
    # Usually local_opt_tour is a permutation where tour[i] connects to tour[i+1]
    # and tour[-1] connects to tour[0].
    # Let's check the length of the tour.
    n = len(local_opt_tour)
    if n > 1:
        # Get distances for all edges in the tour
        # Edges are (tour[0], tour[1]), (tour[1], tour[2]), ..., (tour[n-2], tour[n-1]), (tour[n-1], tour[0])
        src = local_opt_tour[:-1]
        dst = local_opt_tour[1:]
        edge_lens = edge_distance[src, dst]
        
        # Add the closing edge
        closing_len = edge_distance[local_opt_tour[-1], local_opt_tour[0]]
        edge_lens = np.append(edge_lens, closing_len)
        
        avg_edge_len = np.mean(edge_lens)
    else:
        avg_edge_len = 1.0

    # Set the penalty coefficient alpha.
    # We want the penalty for an edge used k times to be alpha * k.
    # If k=5, we want the penalty to be comparable to the edge length itself
    # to significantly alter the 2-opt landscape.
    # Let's target alpha * 5 ~ 0.5 * avg_edge_len, so alpha ~ 0.1 * avg_edge_len.
    # A slightly higher value might help escape deeper local minima.
    alpha = 0.1 * avg_edge_len
    
    # Apply the additive penalty
    # edge_n_used is symmetric, so this will penalize both (i,j) and (j,i)
    perturbed_distance = edge_distance + alpha * edge_n_used
    
    return perturbed_distance
# EVOLVE-BLOCK-END
