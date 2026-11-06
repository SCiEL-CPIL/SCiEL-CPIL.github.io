"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter unvisited nodes that fit in remaining capacity
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Clarke-Wright savings based heuristic
    # For each feasible node, compute a savings score
    # The idea: prefer nodes that are close to current node but also consider
    # the "savings" of not going back to depot
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Savings: how much we save by going to this node instead of going back to depot
    # and then to this node later (approximation)
    # A simple approach: minimize distance from current, but with a bias
    # toward nodes that are also close to the depot (to keep routes compact)
    
    # Combined score: primarily nearest, but penalize nodes far from depot
    # This encourages shorter routes and better consolidation
    alpha = 0.7  # weight on distance from current
    beta = 0.3   # weight on distance to depot
    
    scores = alpha * dist_from_current + beta * dist_to_depot
    
    # Select the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
