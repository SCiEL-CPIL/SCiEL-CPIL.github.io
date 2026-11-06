"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to feasible nodes
    dist_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Lookahead: for each candidate, estimate the cost of going to the next nearest feasible node after it
    # This helps avoid dead-ends where a node is close but leads to a long detour
    n_feasible = len(feasible_nodes)
    
    if n_feasible > 1:
        # For each candidate i, find the nearest other feasible node j (j != i)
        # Score = dist(current, i) + alpha * min(dist(i, j) for j != i)
        # This is a 2-step lookahead
        alpha = 0.3  # Weight for the lookahead term
        
        # dist_matrix between feasible nodes
        feas_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        
        # For each i, the min distance to any other j
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(feas_dist, np.inf)
        
        min_next_dist = np.min(feas_dist, axis=1)
        
        # Total score: immediate distance + weighted lookahead
        scores = dist_to_feasible + alpha * min_next_dist
    else:
        scores = dist_to_feasible
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
