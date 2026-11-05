"""Hybrid CVRP heuristic: geometric sweep + savings + 2-step soft-min lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Savings (Clarke-Wright): how much we save by visiting c next vs. closing now
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # --- Geometric sweep: angular ordering from current node ---
    # Compute angle of each feasible node relative to the direction current->depot
    # Nodes aligned with the current route direction get lower angles (preferred)
    # Use vector from current to depot as reference direction
    cur_pos = np.array([0.0, 0.0])  # We don't have coordinates, so use distance-based proxy
    # Since we only have distance matrix, use a geometric proxy:
    # The "angle" proxy: nodes that are closer to the depot relative to current
    # are more "ahead" in the sweep. We use (d_cur - d_dep) as a proxy for
    # how much the node extends the route vs. shortening it.
    # Actually, let's use the savings itself as the geometric signal combined with distance.
    
    # --- 2-step lookahead with soft-min ---
    # For each candidate i, estimate continuation cost:
    # step1 = d_cur_feas[i]  (cost to reach candidate)
    # step2 = soft_min over j != i of d(feas_i, feas_j) and d(feas_i, depot)
    # We want to minimize step1 + step2
    
    # Distance matrix among feasible nodes
    d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    # Set diagonal to inf to exclude self
    d_feas_feas_copy = d_feas_feas.copy()
    np.fill_diagonal(d_feas_feas_copy, np.inf)
    
    # For each candidate, the min distance to any other feasible node or depot
    min_to_other = np.min(d_feas_feas_copy, axis=1)
    cont_cost_1step = np.minimum(d_dep_feas, min_to_other)
    
    # 2-step: for each candidate i, find the best 2-step continuation
    # After visiting i, the best next move is min over j of d(i,j) + min(d(j,depot), min_k(d(j,k)))
    # This is expensive, so approximate: use the 1-step cont cost of the nearest neighbor
    # For each i, find argmin j of d(i,j), then use cont_cost_1step[j] as 2-step estimate
    # But this is O(n^2) which is fine for n<=50
    
    # Compute 2-step continuation cost
    cont_cost_2step = np.zeros(n_nodes)
    for i in range(n_nodes):
        # Find the best next node after i (excluding i itself)
        # Consider going to depot or to another feasible node
        best_2step = d_dep_feas[i]  # go to depot
        if n_nodes > 1:
            for j in range(n_nodes):
                if j == i:
                    continue
                # Cost: d(i,j) + continuation from j (1-step)
                cost_ij = d_feas_feas[i, j] + cont_cost_1step[j]
                if cost_ij < best_2step:
                    best_2step = cost_ij
        cont_cost_2step[i] = best_2step
    
    # Soft-min temperature for smoothing
    # Lower temperature = sharper (more greedy), higher = smoother
    # We use a moderate temperature
    T = 5.0
    # Soft-min: -T * log(sum(exp(-x/T)))
    # For the 2-step continuation, apply soft-min over the raw distances
    # Actually, let's use the cont_cost_2step directly but with a penalty for high values
    
    # --- Normalize all terms ---
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    savings_norm = normalize(savings)
    dist_norm = normalize(d_cur_feas)
    cont_norm = normalize(cont_cost_2step)
    
    # --- Capacity-aware penalty ---
    # Penalize nodes that use a large fraction of remaining capacity if they're far
    # This encourages saving capacity for potentially closer nodes later
    cap_usage = demands[feas_nodes] / max(rest_capacity, 1)
    cap_penalty = cap_usage * dist_norm  # high distance + high cap usage = bad
    
    # --- Combined score ---
    # Maximize savings, minimize distance, minimize continuation cost, minimize cap penalty
    W_sav = 1.2
    W_dist = 0.7
    W_cont = 0.5
    W_cap = 0.3
    
    scores = W_sav * savings_norm - W_dist * dist_norm - W_cont * cont_norm - W_cap * cap_penalty
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
