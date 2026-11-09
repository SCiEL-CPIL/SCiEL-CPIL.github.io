"""Rewritten program: Robust Lookahead Greedy heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(depot)

    # Filter to capacity-feasible customers.
    feas = unvisited[demands[unvisited] <= rest_capacity]
    if feas.size == 0:
        return int(depot)

    n_feas = feas.size
    if n_feas == 1:
        return int(feas[0])

    # Precompute distances from current to all feasible nodes
    d_cur = distance_matrix[current_node, feas]
    
    best_score = np.inf
    best_node = int(feas[0])
    
    # Iterate through each candidate
    for idx, i in enumerate(feas):
        # Cost to reach i from current
        cost_to_i = d_cur[idx]
        
        # Determine feasible nodes after visiting i
        rem_cap = rest_capacity - demands[i]
        # Feasible next nodes are those in feas (excluding i) that fit in rem_cap
        # We can just check the demands of all feas nodes except i
        mask_next = np.ones(n_feas, dtype=bool)
        mask_next[idx] = False
        next_feas_indices = np.where(mask_next)[0]
        
        if len(next_feas_indices) == 0:
            # No other nodes can be visited, must return to depot
            next_cost = distance_matrix[i, depot]
        else:
            # Check which of the remaining feasible nodes are capacity feasible
            # demands[feas[next_feas_indices]] <= rem_cap
            next_nodes = feas[next_feas_indices]
            next_demands = demands[next_nodes]
            valid_next_mask = next_demands <= rem_cap
            
            if np.any(valid_next_mask):
                valid_next_nodes = next_nodes[valid_next_mask]
                # Find nearest valid next node
                d_to_next = distance_matrix[i, valid_next_nodes]
                next_cost = np.min(d_to_next)
            else:
                # No other nodes fit, return to depot
                next_cost = distance_matrix[i, depot]
        
        total_score = cost_to_i + next_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(i)
            
    return best_node
# EVOLVE-BLOCK-END
