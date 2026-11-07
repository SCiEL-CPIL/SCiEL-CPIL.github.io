"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturb the edge distance matrix to encourage the 2-opt solver to escape
    the current local optimum.
    
    Strategy:
    1. Calculate the current tour length.
    2. Identify edges in the current tour.
    3. Assign penalties to these edges proportional to their usage count,
       scaled such that the total penalty added to the tour is a fraction
       (alpha) of the current tour length.
    4. Return the modified distance matrix.
    """
    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()
    
    # 1. Calculate current tour length
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    if tour_len == 0:
        return edge_distance.copy()
    
    # 2. Identify edges and their usage counts
    edge_usages = []
    edge_indices = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        usage = edge_n_used[u, v]
        edge_usages.append(usage)
        edge_indices.append((u, v))
    
    total_usage = sum(edge_usages)
    
    # 3. Calculate penalties
    # We want the total penalty added to the tour to be alpha * tour_len.
    # Let alpha be 0.2 (20% increase in tour length).
    alpha = 0.2
    target_penalty_sum = alpha * tour_len
    
    if total_usage == 0:
        # If no usage history, apply uniform penalty to all tour edges
        penalty_per_edge = target_penalty_sum / n
        penalties = [penalty_per_edge] * n
    else:
        # Distribute penalty proportionally to usage
        # p_e = (target_penalty_sum * n_e) / total_usage
        penalties = [target_penalty_sum * (usage / total_usage) for usage in edge_usages]
    
    # 4. Create modified distance matrix
    modified_distance = edge_distance.copy()
    
    for i, (u, v) in enumerate(edge_indices):
        p = penalties[i]
        # Ensure penalty is positive and finite
        if p > 0 and np.isfinite(p):
            modified_distance[u, v] += p
            modified_distance[v, u] += p
            
    return modified_distance
# EVOLVE-BLOCK-END
