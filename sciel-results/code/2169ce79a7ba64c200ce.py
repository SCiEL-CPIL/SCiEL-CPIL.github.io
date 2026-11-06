"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Fast path: if no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return 0
    
    # If we cannot fit any more demand, return depot
    if rest_capacity <= 0:
        return 0
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Strategy: combine nearest-neighbor with a "savings" heuristic
    # Score = distance from current + distance to depot (if we go there, how much
    # does it extend the route?) minus a bonus for being close to other unvisited
    # feasible nodes (clustering)
    
    # For clustering: for each candidate, compute mean distance to other feasible nodes
    # This encourages visiting nodes that are close to each other (dense clusters)
    n_feasible = len(feasible_nodes)
    
    # Mean distance from each candidate to all other feasible nodes
    # dist_matrix between feasible nodes
    feas_dist_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Sum of distances from each node to all others (excluding self)
    row_sums = feas_dist_matrix.sum(axis=1) - np.diag(feas_dist_matrix)
    mean_neighbor_dist = row_sums / max(n_feasible - 1, 1)
    
    # Score: lower is better
    # Primary: distance from current node (nearest neighbor)
    # Secondary: distance to depot (prefer nodes that are not too far from depot,
    #   so we don't get stuck far away with little capacity)
    # Tertiary: mean distance to neighbors (prefer dense clusters)
    
    # Normalize components to [0, 1] for weighting
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dist = normalize(dists)
    norm_depot = normalize(dists_to_depot)
    norm_cluster = normalize(mean_neighbor_dist)
    
    # Weighted score
    w1 = 0.5   # proximity to current
    w2 = 0.25  # proximity to depot (to avoid getting stranded)
    w3 = 0.25  # clustering (visit dense areas)
    
    score = w1 * norm_dist + w2 * norm_depot + w3 * norm_cluster
    
    # Pick the node with the lowest score
    best_idx = int(np.argmin(score))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
