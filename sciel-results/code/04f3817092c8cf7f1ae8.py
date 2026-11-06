"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    best_score = float('inf')
    best_node = -1
    
    # Pre-compute distance from current to depot (constant for this step)
    d_current_depot = distance_matrix[current_node, depot]
    
    # Vectorized distance from current to all feasible nodes
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Loop over feasible nodes to calculate lookahead and savings
    for idx, j in enumerate(feasible_nodes):
        d_ij = d_current_feasible[idx]
        
        # Calculate savings: d(i, depot) + d(depot, j) - d(i, j)
        d_depot_j = distance_matrix[depot, j]
        savings = d_current_depot + d_depot_j - d_ij
        
        # Calculate lookahead: min distance from j to next feasible node or depot
        remaining_cap = rest_capacity - demands[j]
        
        # Find nodes that can be visited after j
        # They must be in unvisited, not j, and demand <= remaining_cap
        # Optimization: Create a mask for all unvisited nodes
        # But we need to exclude j.
        
        # We can reuse unvisited_arr
        # Mask for capacity
        cap_mask = demands[unvisited_arr] <= remaining_cap
        # Mask for not being j
        not_j_mask = unvisited_arr != j
        
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        if len(next_feasible_nodes) > 0:
            # Min distance from j to any next feasible node
            d_next = np.min(distance_matrix[j, next_feasible_nodes])
        else:
            # Must go to depot
            d_next = distance_matrix[j, depot]
            
        # Score: Minimize immediate + lookahead, maximize savings
        # Score = d_ij + d_next - lambda * savings
        # Let's use lambda = 1.0
        score = d_ij + d_next - 1.0 * savings
        
        if score < best_score:
            best_score = score
            best_node = j
            
    return int(best_node)
# EVOLVE-BLOCK-END
