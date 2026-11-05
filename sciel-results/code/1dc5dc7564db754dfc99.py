"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour construction.
    Uses a hybrid heuristic:
    - For small remaining sets (k <= 12), uses Dynamic Programming to find the optimal first step.
    - For larger sets, uses a 2-step lookahead greedy heuristic.
    """
    k = len(unvisited_nodes)
    
    if k == 0:
        # Should not happen if called correctly, but return current or dest?
        # Contract implies unvisited_nodes is not empty.
        return int(current_node)

    if k == 1:
        return int(unvisited_nodes[0])

    # Threshold for exact DP solution for the remaining sub-tour's first step
    if k <= 12:
        return _select_next_dp(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _select_next_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)

def _select_next_dp(current_node, destination_node, unvisited_nodes, distance_matrix):
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    # Local distance matrix for unvisited nodes
    D = distance_matrix[np.ix_(U, U)]
    start_costs = distance_matrix[current_node, U]
    end_costs = distance_matrix[U, destination_node]
    
    best_cost = float('inf')
    best_idx = 0
    
    # Try each node as the first step
    for i in range(k):
        # Cost to go from current to U[i]
        cost_to_u_i = start_costs[i]
        
        # Remaining nodes
        R_indices = [j for j in range(k) if j != i]
        m = len(R_indices)
        
        if m == 0:
            # Just go to destination
            total_cost = cost_to_u_i + end_costs[i]
        else:
            # DP for path from U[i] through R to destination
            # Map R to local indices 0..m-1
            # D_R[j][l] is dist between R[j] and R[l]
            D_R = D[np.ix_(R_indices, R_indices)]
            # Start costs for DP: dist from U[i] to R[j]
            start_R = D[i, R_indices]
            # End costs for DP: dist from R[j] to destination
            end_R = end_costs[R_indices]
            
            # DP state: dp[mask][j] = min cost to start at U[i], visit subset 'mask' of R, ending at R[j]
            # mask is an integer bitset of size m
            size = 1 << m
            dp = np.full((size, m), float('inf'))
            
            # Base cases
            for j in range(m):
                dp[1 << j][j] = start_R[j]
                
            # Fill DP table
            for mask in range(1, size):
                # Iterate over ending nodes in this mask
                # Optimization: only iterate j if bit j is set in mask
                for j in range(m):
                    if (mask & (1 << j)) == 0:
                        continue
                    current_cost = dp[mask][j]
                    if current_cost == float('inf'):
                        continue
                    
                    # Try to extend to next node
                    # Find unset bits
                    # Instead of iterating all next, we can iterate all next and check if not in mask
                    for next_j in range(m):
                        if (mask & (1 << next_j)) != 0:
                            continue
                        new_mask = mask | (1 << next_j)
                        new_cost = current_cost + D_R[j, next_j]
                        if new_cost < dp[new_mask][next_j]:
                            dp[new_mask][next_j] = new_cost
            
            # Find min cost to complete the tour
            full_mask = size - 1
            min_path_cost = float('inf')
            for j in range(m):
                if dp[full_mask][j] + end_R[j] < min_path_cost:
                    min_path_cost = dp[full_mask][j] + end_R[j]
            
            total_cost = cost_to_u_i + min_path_cost
            
        if total_cost < best_cost:
            best_cost = total_cost
            best_idx = i
            
    return int(U[best_idx])

def _select_next_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, U]
    
    # Get sub-matrix of distances between unvisited nodes
    D = distance_matrix[np.ix_(U, U)]
    
    # Set diagonal to infinity to exclude self
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)
    
    # Min distance from each node to any other unvisited node (1-step lookahead)
    min_next_dists = np.min(D_copy, axis=1)
    
    # Score: cost to reach node + cost of next step
    scores = dist_from_current + min_next_dists
    
    # If k is small but > 12, we could do better, but 2-step is okay.
    # For k > 12, 2-step lookahead is O(k^2)
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])

# EVOLVE-BLOCK-END
