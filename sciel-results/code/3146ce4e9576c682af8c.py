"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturb the distance matrix by penalizing edges in the current local optimal tour.
    
    Parameters:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour (sequence of node indices).
    edge_n_used: np.ndarray, symmetric count matrix of edge usage.
    
    Returns:
    np.ndarray, modified distance matrix.
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    # Work on a copy to avoid modifying the original
    modified_dist = edge_distance.copy()
    
    # Calculate mean distance for scaling
    # Ensure we only consider finite values if any are inf (though TSP in unit square usually doesn't have inf)
    finite_dists = edge_distance[np.isfinite(edge_distance)]
    if len(finite_dists) > 0:
        mean_dist = np.mean(finite_dists)
    else:
        mean_dist = 1.0
        
    # Hyperparameter for penalty strength
    # Tuned to be significant relative to mean distance
    lambda_penalty = 0.5
    
    # Identify edges in the current tour
    # Tour is a list of nodes [v0, v1, ..., vN-1]
    tour = np.asarray(local_opt_tour)
    n_nodes = len(tour)
    
    if n_nodes > 2:
        for i in range(n_nodes):
            u = tour[i]
            v = tour[(i + 1) % n_nodes]
            
            # Get usage count for this edge
            # edge_n_used is symmetric, so (u,v) is same as (v,u)
            usage = edge_n_used[u, v]
            
            # Calculate penalty
            penalty = lambda_penalty * mean_dist * usage
            
            # Add penalty to the modified distance matrix
            modified_dist[u, v] += penalty
            modified_dist[v, u] += penalty
            
            # Time guard
            if time.time() - t0 > DEADLINE:
                break
                
    return modified_dist
# EVOLVE-BLOCK-END
