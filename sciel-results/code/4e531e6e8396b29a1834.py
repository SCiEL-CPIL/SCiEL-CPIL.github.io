"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # wall-clock safety margin (120s limit + buffer)
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(unvisited_nodes)
    if n == 0:
        return depot

    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: nearest feasible
        feasible = [i for i in range(n) if demands[unvisited_nodes[i]] <= rest_capacity]
        if not feasible:
            return depot
        d = distance_matrix[current_node, unvisited_nodes[feasible]]
        return int(unvisited_nodes[feasible[np.argmin(d)]])

    # Filter feasible nodes (demand fits remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited_nodes[feasible_mask]
    nf = len(feasible_nodes)

    # Distance from current node to each feasible candidate
    dist_from_current = distance_matrix[current_node, feasible_nodes]

    # Distance from depot to each feasible candidate (for savings)
    dist_from_depot = distance_matrix[depot, feasible_nodes]

    # Clarke-Wright savings: visiting i now saves vs. closing route and starting new one
    # savings_i = dist(depot, current) + dist(depot, i) - dist(current, i)
    # Higher savings = better to visit now
    dist_depot_current = distance_matrix[depot, current_node]
    savings = dist_depot_current + dist_from_depot - dist_from_current

    # Look-ahead: 2-step cost if we pick i now
    # cost_2step[i] = dist_from_current[i] + min_j(dist(i, j)) for j != i
    # Lower 2-step cost is better
    if nf > 1:
        pair_dists = np.zeros((nf, nf))
        for idx_i in range(nf):
            pair_dists[idx_i] = distance_matrix[feasible_nodes[idx_i], feasible_nodes]
        np.fill_diagonal(pair_dists, np.inf)
        min_next_dist = np.min(pair_dists, axis=1)  # min dist from i to any other feasible j
        cost_2step = dist_from_current + min_next_dist
    else:
        cost_2step = dist_from_current.copy()

    # Urgency: nodes with high demand relative to remaining capacity are more urgent
    # because they consume more of the limited capacity
    demands_feasible = demands[feasible_nodes]
    urgency = demands_feasible / rest_capacity

    # Combine scores:
    # 1. Savings (higher is better) - prefer nodes that save distance vs. separate trip
    # 2. 2-step cost (lower is better) - prefer nodes that lead to cheaper next steps
    # 3. Urgency (higher is better) - prefer high-demand nodes to fill capacity
    
    # Normalize each component
    savings_range = savings.max() - savings.min() if nf > 1 else 1.0
    cost_2step_range = cost_2step.max() - cost_2step.min() if nf > 1 else 1.0
    urgency_range = urgency.max() - urgency.min() if nf > 1 else 1.0
    
    if savings_range < 1e-9:
        savings_norm = np.zeros(nf)
    else:
        savings_norm = (savings - savings.min()) / savings_range
    
    if cost_2step_range < 1e-9:
        cost_2step_norm = np.zeros(nf)
    else:
        # Invert: lower cost is better, so normalize as 1 - (cost - min)/range
        cost_2step_norm = 1.0 - (cost_2step - cost_2step.min()) / cost_2step_range
    
    if urgency_range < 1e-9:
        urgency_norm = np.zeros(nf)
    else:
        urgency_norm = (urgency - urgency.min()) / urgency_range
    
    # Weighted combination: savings dominates, then 2-step cost, then urgency
    alpha = 0.55  # weight for savings
    beta = 0.30   # weight for 2-step cost
    gamma = 0.15  # weight for urgency
    
    combined = alpha * savings_norm + beta * cost_2step_norm + gamma * urgency_norm
    
    # Pick the candidate with highest combined score
    best_idx = np.argmax(combined)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
