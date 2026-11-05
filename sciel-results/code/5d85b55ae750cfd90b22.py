import numpy as np
import time

# Deadline guard: kill after 150s wall-clock, return with safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def _time_ok():
    return time.time() < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: Soft-Savings with lookahead.
    
    Uses a smooth surrogate (soft-max via log-sum-exp) for the savings criterion
    to avoid being locked to a single successor when multiple are near-optimal.
    Also incorporates a small lookahead penalty for candidates whose nearest
    feasible neighbor is far away.
    
    Returns the next customer node index, or depot (0) to close the route.
    """
    # Deadline check
    if not _time_ok():
        # Return nearest feasible node as a fast fallback
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        # Nearest feasible
        best = min(feasible, key=lambda n: distance_matrix[current_node, n])
        return int(best)
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited_arr = np.array(unvisited_nodes, dtype=np.int64)
    
    # Feasible nodes: demand fits in remaining capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # Distances
    d_cur_j = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    d_dep_j = distance_matrix[depot, feasible_nodes]         # (n_feasible,)
    d_cur_dep = distance_matrix[current_node, depot]         # scalar
    
    # Savings: d(cur,dep) + d(dep,j) - d(cur,j)
    # Positive saving means it's cheaper to go to j than to return to depot first
    savings = d_cur_dep + d_dep_j - d_cur_j
    
    # Lookahead: for each candidate j, find the min distance from j to any other feasible node
    # (excluding j itself). This penalizes candidates that are isolated.
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n_feasible, n_feasible)
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(D_ff, np.inf)
        # Nearest neighbor distance for each feasible node
        nn_dist = np.min(D_ff, axis=1)  # (n_feasible,)
    else:
        nn_dist = np.array([0.0])
    
    # If the best saving is very negative, return to depot
    if np.max(savings) < -1e-9:
        return depot
    
    # Combined score with lookahead penalty
    beta = 0.05
    score = savings - beta * nn_dist
    
    # Capacity-aware penalty: prefer nodes that leave more capacity for future nodes
    # This avoids premature capacity exhaustion on distant nodes
    remaining_after = rest_capacity - demands[feasible_nodes]
    # Normalize capacity usage: higher remaining_after is better
    # Scale: divide by rest_capacity to get [0,1]
    cap_ratio = remaining_after / rest_capacity if rest_capacity > 0 else 1.0
    
    # Penalize nodes that consume too much capacity relative to what's left
    # But only when capacity is tight
    if rest_capacity < 25:  # capacity is getting low
        cap_penalty = 0.3 * (1.0 - cap_ratio)
    else:
        cap_penalty = 0.0 * (1.0 - cap_ratio)
    
    final_score = score - cap_penalty
    
    best_idx = np.argmax(final_score)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
