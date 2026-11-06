import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited <= 15:
        return _solve_held_karp(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _solve_lookahead_nearest(current_node, destination_node, unvisited_nodes, distance_matrix)

def _solve_held_karp(current, dest, unvisited, dist):
    k = len(unvisited)
    if k == 0:
        return dest # Or -1, depends on contract. If no unvisited, we go to dest? 
        # The function returns the NEXT node. If no unvisited, the next node is dest?
        # Usually, the loop stops when unvisited is empty.
        # Let's assume if unvisited is empty, we don't call this.
        
    # Map unvisited to 0..k-1
    # dp[mask][i] = min cost to start at current, visit set mask, end at unvisited[i]
    # mask is integer 0 to 2^k - 1
    # i is 0 to k-1
    
    INF = float('inf')
    # Initialize dp
    # Using list of lists for speed
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    
    # Base cases: start at current, go to unvisited[i]
    for i in range(k):
        dp[1 << i][i] = dist[current, unvisited[i]]
        parent[1 << i][i] = -1 # No previous unvisited node
        
    # Fill DP table
    for mask in range(1 << k):
        for i in range(k):
            if dp[mask][i] == INF:
                continue
            if not (mask & (1 << i)):
                continue
            
            # Try adding j
            remaining = (~mask) & ((1 << k) - 1)
            # Iterate over bits in remaining
            r = remaining
            while r:
                lsb = r & (-r)
                j = lsb.bit_length() - 1
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist[unvisited[i], unvisited[j]]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    parent[new_mask][j] = i
                r -= lsb
                
    full_mask = (1 << k) - 1
    best_cost = INF
    best_last = -1
    for i in range(k):
        cost = dp[full_mask][i] + dist[unvisited[i], dest]
        if cost < best_cost:
            best_cost = cost
            best_last = i
            
    # Backtrack to find the first node
    # The path is current -> ... -> unvisited[first] -> ... -> unvisited[best_last] -> dest
    # We need the first node after current.
    
    # If k == 1, first is 0
    if k == 1:
        return int(unvisited[0])
        
    # Reconstruct path
    path = []
    cur_mask = full_mask
    cur_i = best_last
    while cur_i != -1:
        path.append(cur_i)
        prev_i = parent[cur_mask][cur_i]
        if prev_i == -1:
            break
        cur_mask = cur_mask ^ (1 << cur_i)
        cur_i = prev_i
        
    # path is reversed: [last, ..., first]
    # The first node in the sequence is path[-1]
    first_idx = path[-1]
    return int(unvisited[first_idx])

def _solve_lookahead_nearest(current, dest, unvisited, dist):
    # Improved lookahead: consider 2-step lookahead with better scoring
    # Score = dist[current, u] + 0.5 * min(dist[u, v] for v in unvisited\{u}) + 0.5 * min(dist[u, dest], dist[u, v] for v in unvisited\{u})
    # This balances immediate distance with future connectivity
    
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    best_node = -1
    best_score = float('inf')
    
    # Precompute distance vectors for efficiency
    # dist_current = dist[current, unvisited]
    dist_current = dist[current, unvisited]
    
    for idx_u, u in enumerate(unvisited):
        d1 = dist_current[idx_u]
        
        # Get distances from u to all unvisited nodes
        d_vec = dist[u, unvisited].copy()
        d_vec[idx_u] = float('inf')
        
        # Nearest neighbor distance
        d2 = np.min(d_vec)
        
        # Distance to destination
        d_dest = dist[u, dest]
        
        # Combined score: immediate + future
        # We want to avoid nodes that are far from both their neighbors and the destination
        score = d1 + 0.3 * d2 + 0.2 * d_dest
        
        if score < best_score:
            best_score = score
            best_node = u
            
    return int(best_node)
