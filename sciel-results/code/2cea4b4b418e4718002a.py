"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard (though computation is trivial for N=50)
    # We don't have a global start time, so we assume the call is fast enough.
    # If we were to track global time, we'd need a global variable.
    # For now, the O(N) scan is extremely fast.
    
    if len(unvisited_nodes) == 0:
        return 0

    # 1. Filter feasible nodes: demand[node] <= rest_capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if len(feasible_nodes) == 0:
        return 0

    # 2. Score candidates
    # Heuristic: Minimize (distance from current to node) + (distance from node to depot)
    # This penalizes nodes that are far from the depot, encouraging the vehicle
    # to pick up "local" customers before traveling far away, and to return
    # to the depot when the remaining customers are far.
    
    # Optimization: Use vectorized numpy operations for speed
    feasible_np = np.array(feasible_nodes)
    
    # Distance from current node to all feasible nodes
    dist_current_to_feasible = distance_matrix[current_node, feasible_np]
    
    # Distance from depot to all feasible nodes
    dist_depot_to_feasible = distance_matrix[depot, feasible_np]
    
    # Score: sum of distances
    scores = dist_current_to_feasible + dist_depot_to_feasible
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(feasible_np[min_idx])
# EVOLVE-BLOCK-END
