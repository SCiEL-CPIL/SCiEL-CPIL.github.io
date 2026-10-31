import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor to ensure completion
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        # Compare cost of picking c1 then c2 vs picking c2 then c1
        # Path: current -> c1 -> c2 -> dest
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        # Path: current -> c2 -> c1 -> dest
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost1 <= cost2:
            return int(c1)
        else:
            return int(c2)

    # For n_unvisited > 2, use 3-step lookahead with destination bias
    # Score(c) = d(u, c) + min_{k != c} ( d(c, k) + min_{m != c, k} ( d(k, m) + 0.1 * d(m, dest) ) )
    
    # Precompute distances from destination to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Create submatrix of distances among unvisited nodes
    # unvisited_nodes is an array of indices
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We need to calculate min_{m != c, k} (d(k, m) + 0.1 * d(m, dest)) for each k, excluding c
    # This is slightly complex to vectorize perfectly without loops in Python, 
    # but N=50 is small.
    
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    # Loop over candidates c
    for i, c in enumerate(unvisited_nodes):
        # Distance from current to c
        cost_current_to_c = distance_matrix[current_node, c]
        
        if cost_current_to_c >= best_cost:
            continue
            
        # Find best k != c
        # k is index j in unvisited_nodes where j != i
        min_future_cost = np.inf
        
        # Loop over k
        for j, k in enumerate(unvisited_nodes):
            if j == i:
                continue
            
            cost_c_to_k = distance_matrix[c, k]
            
            if cost_current_to_c + cost_c_to_k >= best_cost:
                continue
                
            # Find best m != c, k
            # m is index l in unvisited_nodes where l != i and l != j
            min_m_cost = np.inf
            for l, m in enumerate(unvisited_nodes):
                if l == i or l == j:
                    continue
                
                cost_k_to_m = distance_matrix[k, m]
                # Add destination bias
                total_m_cost = cost_k_to_m + 0.1 * dist_to_dest[l]
                
                if total_m_cost < min_m_cost:
                    min_m_cost = total_m_cost
            
            total_candidate_cost = cost_current_to_c + cost_c_to_k + min_m_cost
            
            if total_candidate_cost < min_future_cost:
                min_future_cost = total_candidate_cost
                
        total_cost = cost_current_to_c + min_future_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = c
            
    return int(best_node)
