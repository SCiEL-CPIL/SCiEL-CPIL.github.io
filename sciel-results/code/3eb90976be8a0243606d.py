import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a 2-step lookahead heuristic.
    For each candidate, it estimates the cost of visiting it and then the best
    subsequent node, aiming to minimize the total path segment cost.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes)
    demands_arr = np.array(demands)
    dm = np.array(distance_matrix)
    
    node_demands = demands_arr[unvisited_arr]
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
        
    feasible_nodes = unvisited_arr[feasible_mask]
    
    best_score = float('inf')
    best_node = int(depot)
    
    for j in feasible_nodes:
        # Cost to reach j
        cost_to_j = dm[current_node, j]
        
        # Check if we can visit another node after j
        # Remaining capacity after j
        rem_cap_after_j = rest_capacity - demands_arr[j]
        
        # Find feasible nodes after j
        # We need to exclude j from unvisited for this lookahead
        # Since we are just estimating, we can check all unvisited nodes != j
        # and check capacity
        
        # If no capacity left, we must return to depot
        if rem_cap_after_j <= 0:
            total_score = cost_to_j + dm[j, depot]
        else:
            # Find best next node k
            # Candidates are unvisited nodes != j with demand <= rem_cap_after_j
            # We can vectorize this inner loop for speed, or just loop since N is small
            
            # Get indices of unvisited nodes
            # We need to be careful: 'unvisited_arr' contains j.
            # We want to pick k from unvisited_arr where k != j and demand[k] <= rem_cap_after_j
            
            # Create mask for k
            # We can just iterate or use numpy
            # Let's use numpy for the inner step too
            
            # Indices of unvisited nodes
            # We need to map back to original indices? 
            # unvisited_arr contains the node IDs.
            
            # Filter unvisited_arr for k != j
            k_candidates_mask = (unvisited_arr != j) & (demands_arr[unvisited_arr] <= rem_cap_after_j)
            
            if np.any(k_candidates_mask):
                k_candidates = unvisited_arr[k_candidates_mask]
                
                # Cost from j to k + k to depot
                # d(j, k) + d(k, depot)
                d_jk = dm[j, k_candidates]
                d_kd = dm[k_candidates, depot]
                
                # Minimize d_jk + d_kd
                min_k_cost = np.min(d_jk + d_kd)
                total_score = cost_to_j + min_k_cost
            else:
                # No further nodes possible, return to depot
                total_score = cost_to_j + dm[j, depot]
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(j)
            
    return best_node
