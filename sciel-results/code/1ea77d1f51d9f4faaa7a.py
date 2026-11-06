import numpy as np
import time

_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Use Held-Karp DP to find the best next node for small unvisited sets."""
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to indices 0..n-1
    node_list = list(unvisited_nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # DP: dp[mask][i] = min cost to visit all nodes in mask, ending at node i
    # mask is a bitmask over n nodes
    full_mask = (1 << n) - 1
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Base case: start at current_node, go to any unvisited node
    for j in range(n):
        dp[1 << j][j] = distance_matrix[current_node, node_list[j]]
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == float('inf'):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + distance_matrix[node_list[i], node_list[j]]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
    
    # Find the best last node before returning to destination
    best_cost = float('inf')
    best_last = -1
    for j in range(n):
        total = dp[full_mask][j] + distance_matrix[node_list[j], destination_node]
        if total < best_cost:
            best_cost = total
            best_last = j
    
    if best_last == -1:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # Reconstruct the first node in the optimal path
    # We need to find which node was visited first
    # Backtrack from the full mask
    mask = full_mask
    last = best_last
    first_node = -1
    for step in range(n - 1, -1, -1):
        if step == 0:
            first_node = last
            break
        for prev in range(n):
            if not (mask & (1 << prev)):
                continue
            prev_mask = mask & ~(1 << last)
            if dp[prev_mask][prev] + distance_matrix[node_list[prev], node_list[last]] == dp[mask][last]:
                first_node = prev if step == 1 else first_node
                last = prev
                mask = prev_mask
                break
    
    if first_node == -1:
        # Fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    return int(node_list[first_node])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're running low on time, fall back to simple nearest neighbor
    if time.time() > _DEADLINE - 2.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # For small unvisited sets, use exact Held-Karp
    if n_unvisited <= 12:
        return _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use a combined heuristic:
    # 1. Compute distances
    dist_v_to_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_v_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_v = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Multi-step lookahead (3 steps as before)
    M_temp = dist_v_to_unvisited + dist_v_to_dest[None, :]
    np.fill_diagonal(M_temp, np.inf)
    best_cost_from_w = np.min(M_temp, axis=1)
    
    M2 = dist_v_to_unvisited + best_cost_from_w[None, :]
    np.fill_diagonal(M2, np.inf)
    best_cost_from_v = np.min(M2, axis=1)
    
    total_cost_lookahead = dist_current_to_v + best_cost_from_v
    
    # 3. Return-aware nearest neighbor: penalize nodes far from destination
    # Scale distances by a factor that encourages returning to destination
    # Use a simple heuristic: cost = dist(cur, v) + alpha * dist(v, dest)
    # where alpha is chosen to balance exploration and return
    alpha = 0.5  # Can be tuned
    total_cost_nn = dist_current_to_v + alpha * dist_v_to_dest
    
    # 4. Combine: use the lookahead cost primarily, but also consider NN
    # Weighted combination
    weight_lookahead = 0.7
    weight_nn = 0.3
    total_cost = weight_lookahead * total_cost_lookahead + weight_nn * total_cost_nn
    
    # 5. Pick the best
    best_idx = np.argmin(total_cost)
    return int(unvisited_nodes[best_idx])
