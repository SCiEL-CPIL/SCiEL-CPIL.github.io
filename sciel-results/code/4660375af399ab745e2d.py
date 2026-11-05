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
    
    # 1-step savings: d(cur,dep) + d(dep,j) - d(cur,j)
    savings_1 = d_cur_dep + d_dep_j - d_cur_j
    
    # 2-step lookahead: for each candidate j, estimate the best continuation saving
    # Saving(j->k) = d(j,dep) + d(dep,k) - d(j,k)
    # For fixed j, maximize over k != j: d(j,dep) + (d(dep,k) - d(j,k))
    # Since d(j,dep) is constant for j, maximize (d(dep,k) - d(j,k))
    if n_feasible > 1:
        D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        d_dep_all = distance_matrix[depot, feasible_nodes]
        
        # For each j, compute savings to all other k: d(j,dep) + d(dep,k) - d(j,k)
        # d(j,dep) is d_dep_all[j] (broadcast along axis 1)
        # d(dep,k) is d_dep_all (broadcast along axis 0)
        # d(j,k) is D_ff
        savings_2d = d_dep_all[:, np.newaxis] + d_dep_all[np.newaxis, :] - D_ff
        # Set diagonal to -inf so we don't pick self
        np.fill_diagonal(savings_2d, -np.inf)
        # Best 2-step saving from each j
        best_saving_2 = np.max(savings_2d, axis=1)
        
        # Also compute nearest neighbor distance for soft penalty
        D_ff_nn = D_ff.copy()
        np.fill_diagonal(D_ff_nn, np.inf)
        nn_dist = np.min(D_ff_nn, axis=1)
        
        # Soft-min of distances to other feasible nodes
        soft_min_temp = 2.0
        neg_dist_over_T = -D_ff_nn / soft_min_temp
        neg_max = np.max(neg_dist_over_T, axis=1)
        soft_min_vals = -soft_min_temp * np.log(np.sum(np.exp(neg_dist_over_T - neg_max), axis=1))
        soft_min_vals = np.where(np.isinf(soft_min_vals), 0.0, soft_min_vals)
    else:
        best_saving_2 = np.array([0.0])
        soft_min_vals = np.array([0.0])
    
    # Capacity ratio
    capacity_ratio = demands[feasible_nodes] / max(rest_capacity, 1)
    
    # Combined score: 1-step savings + weighted 2-step lookahead - penalties
    # The 2-step term rewards nodes that have good continuations
    beta_2step = 0.25  # weight for 2-step lookahead
    beta_nn = 0.02     # weight for nearest-neighbor penalty
    beta_cap = 0.12    # weight for capacity consumption penalty
    
    # Normalize 2-step savings to comparable scale
    s2_norm = best_saving_2 - np.min(best_saving_2)
    s2_range = max(np.max(s2_norm), 1e-9)
    s2_frac = s2_norm / s2_range  # 0 to 1
    
    # 1-step savings normalization for adaptive capacity penalty
    savings_norm = savings_1 - np.min(savings_1)
    savings_range = max(np.max(savings_norm), 1e-9)
    savings_frac = savings_norm / savings_range  # 0 to 1
    
    # Adaptive capacity penalty
    adaptive_cap_weight = beta_cap * (1.0 - 0.5 * savings_frac)
    
    # Final score: immediate savings + continuation potential - isolation/capacity penalties
    score = savings_1 + beta_2step * s2_frac * np.max(savings_1) - beta_nn * soft_min_vals - adaptive_cap_weight * capacity_ratio * np.max(savings_1)
    
    # If the best 1-step saving is very negative, return to depot
    if np.max(savings_1) < -1e-9:
        return depot
    
    # Select the node with the highest combined score
    best_idx = np.argmax(score)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
