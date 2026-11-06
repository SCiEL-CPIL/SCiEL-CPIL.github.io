# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # For small n, use Held-Karp exact DP
    if n_unvisited <= 12:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger n, use greedy + 2-opt for each candidate
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    for cand in unvisited_nodes:
        # Build greedy path from cand through remaining
        remaining = np.delete(unvisited_nodes, np.where(unvisited_nodes == cand)[0])
        curr = cand
        path = [cand]
        total_cost = distance_matrix[current_node, cand]
        
        while len(remaining) > 0:
            dists = distance_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            total_cost += dists[min_idx]
            remaining = np.delete(remaining, min_idx)
            curr = next_node
            path.append(curr)
            if total_cost >= best_cost:
                break
        
        if len(remaining) == 0:
            total_cost += distance_matrix[curr, destination_node]
            # Apply 2-opt improvement
            improved_cost = _two_opt_improve(path, distance_matrix, current_node, destination_node)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_node = cand
    
    return int(best_node)

def _two_opt_improve(path, distance_matrix, start_node, end_node):
    """Apply 2-opt with O(n) per pass using vectorized delta computation."""
    n = len(path)
    if n < 2:
        return distance_matrix[start_node, path[0]] + distance_matrix[path[0], end_node]
    
    p = list(path)
    current_cost = distance_matrix[start_node, p[0]]
    for i in range(n - 1):
        current_cost += distance_matrix[p[i], p[i + 1]]
    current_cost += distance_matrix[p[n - 1], end_node]
    
    improved = True
    max_passes = 15
    
    for _pass in range(max_passes):
        if not improved:
            break
        improved = False
        
        best_delta = -1e-10
        best_i = -1
        best_j = -1
        
        for i in range(n - 1):
            # For each i, find the best j > i
            # Old edges: (p[i], p[i+1]) and (p[j], p[j+1] or end)
            # New edges: (p[i], p[j]) and (p[i+1], p[j+1] or end)
            
            old_e1 = distance_matrix[p[i], p[i + 1]]
            
            # Vectorize over j
            js = np.arange(i + 1, n)
            # p[j] indices
            pj = np.array([p[j] for j in js])
            # p[j+1] or end
            pj1 = np.where(js == n - 1, end_node, np.array([p[j + 1] for j in js]))
            
            new_e1 = distance_matrix[p[i], pj]
            new_e2 = distance_matrix[p[i + 1], pj1]
            old_e2 = np.where(js == n - 1,
                             distance_matrix[p[n - 1], end_node],
                             np.array([distance_matrix[p[j], p[j + 1]] for j in js]))
            
            deltas = new_e1 + new_e2 - old_e1 - old_e2
            min_idx = np.argmin(deltas)
            
            if deltas[min_idx] < best_delta:
                best_delta = deltas[min_idx]
                best_i = i
                best_j = js[min_idx]
        
        if best_delta < -1e-10:
            # Perform reversal p[best_i+1..best_j]
            p[best_i + 1:best_j + 1] = p[best_i + 1:best_j + 1][::-1]
            current_cost += best_delta
            improved = True
    
    return current_cost

def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Use Held-Karp DP to find the optimal next node."""
    n = len(unvisited_nodes)
    node_to_idx = {node: i for i, node in enumerate(unvisited_nodes)}
    
    sub_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_dist[i, j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    
    full_mask = (1 << n) - 1
    dp = [[np.inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    for j in range(n):
        dp[1 << j][j] = distance_matrix[current_node, unvisited_nodes[j]]
    
    for S in range(1, 1 << n):
        for j in range(n):
            if not (S & (1 << j)):
                continue
            if dp[S][j] == np.inf:
                continue
            for k in range(n):
                if S & (1 << k):
                    continue
                new_S = S | (1 << k)
                new_cost = dp[S][j] + sub_dist[j, k]
                if new_cost < dp[new_S][k]:
                    dp[new_S][k] = new_cost
                    parent[new_S][k] = j
    
    best_total = np.inf
    best_last = -1
    for j in range(n):
        total = dp[full_mask][j] + distance_matrix[unvisited_nodes[j], destination_node]
        if total < best_total:
            best_total = total
            best_last = j
    
    S = full_mask
    j = best_last
    while parent[S][j] != -1:
        prev = parent[S][j]
        S = S & ~(1 << j)
        j = prev
    
    return int(unvisited_nodes[j])
# EVOLVE-BLOCK-END
