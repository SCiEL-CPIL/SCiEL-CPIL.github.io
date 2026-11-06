"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot

    unvisited_arr = np.array(unvisited_nodes)
    unvisited_demands = demands[unvisited_arr]
    
    # Filter by capacity
    feasible_mask = unvisited_demands <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    best_score = float('inf')
    best_node = -1
    
    # Distances from current to feasible nodes
    d_from_current = distance_matrix[current_node, feasible_nodes]
    # Distances from feasible nodes to depot
    d_to_depot = distance_matrix[feasible_nodes, depot]
    # Demands of feasible nodes
    d_feasible = demands[feasible_nodes]
    
    # Precompute distances from all unvisited nodes to depot for lookahead
    d_unvisited_to_depot = distance_matrix[unvisited_arr, depot]

    for i in range(len(feasible_nodes)):
        j = feasible_nodes[i]
        imm_cost = d_from_current[i]
        ret_cost = d_to_depot[i]
        
        new_cap = rest_capacity - d_feasible[i]
        
        # Find best lookahead node k
        # k in unvisited_arr, k != j, demands[k] <= new_cap
        mask = (unvisited_demands <= new_cap).copy()
        # Exclude j
        mask[unvisited_arr == j] = False
        
        if np.any(mask):
            d_j_to_unvisited = distance_matrix[j, unvisited_arr]
            costs = d_j_to_unvisited + d_unvisited_to_depot
            min_lookahead = np.min(costs[mask])
        else:
            min_lookahead = float('inf')
            
        future_cost = min(ret_cost, min_lookahead)
        total_score = imm_cost + future_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = j
            
    return int(best_node)

# EVOLVE-BLOCK-END
