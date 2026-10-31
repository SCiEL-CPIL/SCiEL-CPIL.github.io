"""
TSP constructive heuristic: Hybrid Regret-2 with MST-awareness and minimax penalty.

Combines:
- Regret-2 structure from the current program (opportunity-cost based selection)
- MST-awareness from Inspiration 1 (global connectivity cost of remaining nodes)
- Minimax centrality penalty from Inspiration 1 (avoid isolated nodes)

Strategy:
For each candidate u in unvisited_nodes, compute:
1. Immediate cost: d(current, u)
2. Regret-2 term: c2(u) - c1(u) where c1 = d(current, u), 
   c2 = min_{v != u} (d(u, v) + d(v, destination))
   This captures opportunity cost.
3. MST weight of remaining nodes (excluding u): captures residual connectivity
4. Minimax penalty: max_{v in unvisited} d(u, v) - avoids isolated nodes

Final score = d(current, u) + alpha * regret(u) + beta * MST(remaining) + gamma * minimax(u)
Minimize this score to select the next node.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Uses Prim's algorithm. For N=50, this is O(N^2) which is acceptable.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Prim's algorithm
    dist_to_mst = [float('inf')] * n
    dist_to_mst[0] = 0.0
    visited = [False] * n
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find unvisited node with min dist_to_mst
        min_val = float('inf')
        u = -1
        for i in range(n):
            if not visited[i] and dist_to_mst[i] < min_val:
                min_val = dist_to_mst[i]
                u = i
        
        if u == -1:
            break  # Should not happen in complete graph
            
        visited[u] = True
        total_weight += min_val
        
        # Update neighbors
        u_global = nodes[u]
        for v in range(n):
            if not visited[v]:
                v_global = nodes[v]
                d = distance_matrix[u_global, v_global]
                if d < dist_to_mst[v]:
                    dist_to_mst[v] = d
                    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Hybrid Regret-2 with MST-awareness and minimax penalty.
    For each candidate node u in unvisited_nodes, compute a composite score:
    score(u) = d(current, u) 
              + alpha * regret(u) 
              + beta * MST_weight(remaining_nodes_excluding_u)
              + gamma * max_{v in unvisited} d(u, v)
    
    Where:
    - regret(u) = c2(u) - c1(u), with c1 = d(current, u),
      c2 = min_{v != u} (d(u, v) + d(v, destination))
    - MST_weight captures the residual connectivity cost
    - max distance penalizes nodes that are far from others (isolation risk)
    
    We select the node with the minimum score.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # Exact 2-node case: compare both orderings
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute regret-2 term
    # For each u, c2(u) = min_{v != u} (d(u, v) + d(v, destination))
    future_costs = d_sub + d_dest[None, :]
    np.fill_diagonal(future_costs, np.inf)
    c2 = np.min(future_costs, axis=1)
    regret = c2 - d_curr  # c2(u) - c1(u)
    
    # 5. Compute minimax term: max distance from each u to any other v
    d_sub_for_max = d_sub.copy()
    np.fill_diagonal(d_sub_for_max, 0)
    max_dist_to_other = np.max(d_sub_for_max, axis=1)
    
    # 6. Weights for the hybrid score
    # Tuned to balance immediate cost, regret, MST, and minimax
    alpha = 0.3   # Weight for regret term
    beta = 0.5    # Weight for MST term
    gamma = 0.1   # Weight for minimax term
    
    # 7. Compute MST weights for remaining nodes (excluding each candidate)
    # This is the most expensive part, but N=50 makes it feasible
    mst_weights = np.zeros(n)
    for i in range(n):
        remaining_mask = np.ones(n, dtype=bool)
        remaining_mask[i] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        if len(remaining_nodes) == 0:
            mst_weights[i] = 0.0
        else:
            mst_weights[i] = _mst_weight(remaining_nodes, distance_matrix)
        
        # Check deadline during MST computation
        if _is_deadline_passed():
            # Fallback: use precomputed distances for a quick decision
            dists = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(dists)])
    
    # 8. Compute final composite score
    scores = d_curr + alpha * regret + beta * mst_weights + gamma * max_dist_to_other
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
