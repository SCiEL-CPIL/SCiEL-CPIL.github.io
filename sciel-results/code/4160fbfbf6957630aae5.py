"""Rewritten program: TSP constructive heuristic using regret-based savings with MST-awareness."""
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

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Regret-based heuristic with MST-neighbor awareness.
    
    For each candidate node u in unvisited_nodes, we compute:
    1. Immediate cost: d(current, u)
    2. Regret: how much worse the best alternative would be if we skip u
       - For each other candidate v, compute the cost of going current->v->u vs current->u->v
       - The regret of u is the maximum over v of (cost_delayed(u) - cost_now(u))
       - High regret means u should be visited soon or we'll pay a penalty
    3. MST-awareness: nodes well-connected to the unvisited set (low average
       distance to other unvisited nodes) are preferred to avoid creating
       long edges that are hard to fix later
    
    Combined score: score(u) = d(current, u) + lambda_regret * regret(u) + lambda_mst * mst_penalty(u)
    where mst_penalty(u) = avg_dist(u, unvisited) - min_avg_dist (normalized)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # For 2 nodes, pick the one that minimizes the total path
        # current -> u -> v -> destination
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute regret for each candidate
    # For candidate i, the regret is:
    # max over j != i of [d(current, j) + d(j, i) - d(current, i)]
    # This represents the extra cost if we visit j before i instead of i before j
    # (assuming we'd go current -> j -> i vs current -> i -> j)
    
    # For each pair (i, j), the "delay cost" of visiting j before i is:
    # delay_cost[i, j] = d(current, j) + d(j, i) - d(current, i)
    # This can be negative (meaning visiting j before i is actually better)
    
    # Build the delay cost matrix
    # delay_cost[i, j] = d_curr[j] + d_sub[j, i] - d_curr[i]
    # = d_curr[j] - d_curr[i] + d_sub[j, i]
    
    # d_curr[j] for all j, broadcast along axis 0
    # d_curr[i] for all i, broadcast along axis 1
    # d_sub[j, i] is the transpose of d_sub[i, j]
    
    delay_cost = d_curr[np.newaxis, :] - d_curr[:, np.newaxis] + d_sub.T  # shape (n, n), [i, j]
    
    # Set diagonal to -inf (we don't consider i == j)
    np.fill_diagonal(delay_cost, -np.inf)
    
    # Regret for candidate i is the max over j != i of delay_cost[i, j]
    # But we only care about positive regret (if all are negative, regret is 0)
    regret = np.maximum(0.0, np.max(delay_cost, axis=1))
    
    # 5. MST-awareness: compute average distance from each node to other unvisited nodes
    # Low average distance means the node is well-connected (good for MST-like tours)
    np.fill_diagonal(d_sub, 0)  # Don't count self-distance
    avg_dist = np.mean(d_sub, axis=1)
    
    # Normalize: subtract the minimum so the best-connected node has penalty 0
    min_avg = np.min(avg_dist)
    mst_penalty = avg_dist - min_avg
    
    # 6. Adaptive weights based on remaining nodes
    remaining_fraction = n / 50.0
    
    # Regret becomes more important as we go (we don't want to leave bad options for last)
    lambda_regret = 0.5 * (1.0 + remaining_fraction)
    
    # MST-awareness is more important early (to establish good structure)
    lambda_mst = 0.3 * remaining_fraction
    
    # 7. Combine all terms
    scores = d_curr + lambda_regret * regret + lambda_mst * mst_penalty
    
    # 8. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
