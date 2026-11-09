"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Filter feasible nodes
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Distances
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # Heuristic Selection
    if current_node == depot:
        # Start with nearest neighbor to establish a direction
        best_idx = np.argmin(dists_current)
    else:
        # Savings-based heuristic: Minimize (dist to current - alpha * dist to depot)
        # This favors nodes that are close to current but far from depot,
        # encouraging long, efficient routes.
        alpha = 0.9
        scores = dists_current - alpha * dists_depot
        best_idx = np.argmin(scores)
        
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
