"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: pick next customer or depot (0) to close route.
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node by:
         - immediate distance from current_node
         - plus a lookahead: minimum distance from that node to any other feasible node (or depot if it's the last)
         This encourages picking nodes that keep the route short overall.
      4. Use a weighted combination to balance proximity and connectivity.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Immediate distances from current node
    immediate_dists = distance_matrix[current_node, feasible_nodes]
    
    # Lookahead: for each candidate, compute min distance to any other feasible node
    # or to depot (if it could be the last)
    # To keep it fast, we compute min distance from each candidate to all other feasible nodes
    # and also consider distance to depot as an option for closing
    
    # Build submatrix of feasible_nodes x feasible_nodes distances
    feas_idx = feasible_nodes
    sub_dist = distance_matrix[np.ix_(feas_idx, feas_idx)]
    
    # Min distance from each node to any other feasible node (excluding self)
    np.fill_diagonal(sub_dist, np.inf)
    min_other_dist = np.min(sub_dist, axis=1)
    
    # Distance to depot
    depot_dist = distance_matrix[feasible_nodes, depot]
    
    # Combined score: immediate + alpha * (min_other_dist + beta * depot_dist)
    # We want to minimize total route length, so we prefer:
    # - Short immediate distance
    # - Nodes that are close to other nodes (to continue route efficiently)
    # - If it's likely the last node, being close to depot helps
    
    # Weight immediate distance heavily, add a small lookahead term
    alpha = 0.3
    beta = 0.5
    
    lookahead = alpha * (min_other_dist + beta * depot_dist)
    total_score = immediate_dists + lookahead
    
    # Pick the node with minimum total score
    best_idx = np.argmin(total_score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
