import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Approach: Multiplicative Penalty + Additive Kick for Current Tour
    # The multiplicative penalty scales with the edge's own length,
    # which preserves the relative quality of short edges better than
    # a global additive penalty.
    
    # Scaling factor for the multiplicative penalty.
    # An edge used 'k' times gets its cost increased by 'alpha * k' factor.
    alpha = 0.15
    
    # Base penalized distance: c'_ij = c_ij * (1 + alpha * n_ij)
    # This is equivalent to c_ij + alpha * n_ij * c_ij
    penalized_dist = edge_distance * (1.0 + alpha * edge_n_used)
    
    # Apply an additional additive penalty to the edges in the current local optimum.
    # This "kick" forces the 2-opt step to consider breaking the current tour structure.
    mean_dist = np.mean(edge_distance)
    kick_penalty = 0.2 * mean_dist
    
    n = len(local_opt_tour)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalized_dist[u, v] += kick_penalty
        penalized_dist[v, u] += kick_penalty
        
    return penalized_dist
