"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = None
_DEADLINE = 115.0  # 5s margin before 120s

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for fast indexing
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    n_unvisited = len(unvisited)
    
    # Filter by capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited], dtype=bool)
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 0:
        return depot

    # If only one feasible node, go there
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Time check: if we're running out of time, use simple nearest neighbor
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # Get distances
    d_current = distance_matrix[current_node, feasible_nodes]
    d_depot = distance_matrix[depot, feasible_nodes]
    d_depot_current = distance_matrix[depot, current_node]
    
    # Savings score: d(depot, current) + d(depot, j) - d(current, j)
    # Higher savings is better (we want to maximize savings, i.e., minimize negative savings)
    savings = d_depot_current + d_depot - d_current
    neg_savings = -savings
    
    # Soft-min lookahead: smooth surrogate for min distance to any other feasible node or depot
    # This penalizes dead-ends (nodes far from everything else)
    indices = np.concatenate([feasible_nodes, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]
    
    t = 1.5  # temperature for soft-min
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9
    
    # Numerical stability
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    s = np.maximum(s, 1e-300)
    lookahead = row_min[:n_feasible] + t * np.log(s[:n_feasible])
    
    # Capacity utilization bonus: prefer nodes that use more remaining capacity
    cap_bonus = -0.12 * (demands[feasible_nodes] / rest_capacity)
    
    # Proximity tie-break
    dist_min = np.min(d_current)
    dist_max = np.max(d_current)
    dist_range = dist_max - dist_min + 1e-10
    dist_norm = (d_current - dist_min) / dist_range
    
    # Combined score (minimize)
    # Weights: savings dominant, lookahead secondary, cap bonus and proximity small
    alpha_savings = 1.0
    alpha_lookahead = 0.35
    alpha_cap = 1.0
    alpha_dist = 0.15
    
    scores = (alpha_savings * neg_savings
              + alpha_lookahead * lookahead
              + alpha_cap * cap_bonus
              + alpha_dist * dist_norm)
    
    # Pick the one with minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
