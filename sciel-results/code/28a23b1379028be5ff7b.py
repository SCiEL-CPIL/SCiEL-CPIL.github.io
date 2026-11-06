"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Threshold for switching to exact DP
DP_THRESHOLD = 15

def _solve_greedy(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Heuristic for large sets of unvisited nodes.
    Uses a 2-step lookahead: d(curr, cand) + d(cand, dest) + 0.5 * min_next_dist
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # d_curr: distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    # d_dest: distances from all unvisited to destination
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Calculate min distance from each candidate to any other unvisited node
    # d_sub is a matrix of distances between unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row (candidate), find min distance to others
    # Set diagonal to inf so min doesn't pick self
    np.fill_diagonal(d_sub, np.inf)
    min_next = np.min(d_sub, axis=1)
    
    # If there's only 1 unvisited, min_next will be inf, but we handled n==1 above.
    # If n==2, min_next is just the distance to the other one.
    
    scores = d_curr + d_dest + 0.5 * min_next
    
    min_idx = np.argmin(scores)
    return int(unvisited_nodes[min_idx])

def _solve_exact_suffix(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Exact DP (Held-Karp style) for small sets of unvisited nodes.
    Finds the node that minimizes the total tour cost for the suffix.
    """
    k = len(unvisited_nodes)
    if k == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to local indices 0..k-1
    # unvisited_nodes is a numpy array of global indices
    
    # Precompute distances between unvisited nodes and to/from current/dest
    # d_unvisited[i][j] = distance between unvisited[i] and unvisited[j]
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # d_to_dest[i] = distance from unvisited[i] to destination
    d_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # d_from_curr[i] = distance from current to unvisited[i]
    d_from_curr = distance_matrix[current_node, unvisited_nodes]
    
    # DP Table: dp[mask][last]
    # mask: bitmask of nodes in unvisited_nodes that are "remaining" to be visited
    #        starting from 'last' and ending at destination.
    #        Wait, let's stick to the definition:
    #        g[mask][last] = min cost to start at 'last', visit all nodes in 'mask', and end at destination.
    #        
    #        Base case: mask = 0. g[0][last] = d(last, destination) = d_to_dest[last]
    #        Recurrence: g[mask][last] = min_{j in mask} ( d(last, unvisited[j]) + g[mask without j][j] )
    #        
    #        We want to find the best first node v.
    #        Total cost if v is first = d(current, v) + g[AllMask without v][v]
    
    size = 1 << k
    # Initialize dp with large values
    dp = [[float('inf')] * k for _ in range(size)]
    
    # Base case: no nodes left to visit in mask
    for i in range(k):
        dp[0][i] = d_to_dest[i]
        
    # Fill DP table
    # Iterate over all masks
    for mask in range(1, size):
        for last in range(k):
            # 'last' is the node we are currently at.
            # We need to choose the next node 'j' from 'mask'.
            # Wait, the recurrence is:
            # g[mask][last] = min_{j in mask} ( d(last, j) + g[mask ^ (1<<j)][j] )
            # This means we move from 'last' to 'j', and then solve for the rest of 'mask' excluding 'j' starting from 'j'.
            
            # Optimization: if 'last' is not in the "context" of mask?
            # Actually, 'last' is the start of this subproblem. It doesn't need to be in 'mask'.
            # 'mask' contains the nodes we must visit *after* leaving 'last'.
            
            min_cost = float('inf')
            # Iterate through all bits set in mask
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                j = int(np.log2(lsb)) # index of the bit
                # cost to go from last to j
                cost = d_unvisited[last][j] + dp[mask ^ lsb][j]
                if cost < min_cost:
                    min_cost = cost
                m ^= lsb
            
            dp[mask][last] = min_cost
            
    # Find the best first node
    all_mask = size - 1
    best_cost = float('inf')
    best_node_idx = -1
    
    for i in range(k):
        # If we pick unvisited[i] first:
        # Cost = d(current, unvisited[i]) + g[all_mask without i][i]
        # The mask for the suffix is all_mask with bit i cleared
        suffix_mask = all_mask ^ (1 << i)
        total_cost = d_from_curr[i] + dp[suffix_mask][i]
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node_idx = i
            
    return int(unvisited_nodes[best_node_idx])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    start_time = time.time()
    
    # Deadline guard
    # If we are running low on time, fall back to simple nearest neighbor
    if time.time() - start_time > 110:
         d_curr = distance_matrix[current_node, unvisited_nodes]
         return int(unvisited_nodes[np.argmin(d_curr)])
         
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    if len(unvisited_nodes) <= DP_THRESHOLD:
        return _solve_exact_suffix(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _solve_greedy(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
