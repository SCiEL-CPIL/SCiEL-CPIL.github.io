"""Combined CVRP heuristic: geometric sweep + savings + normalized lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings (Clarke-Wright)
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Geometric sweep term: approximate angular alignment using triangle inequality proxy.
    # A node is "aligned" with the current heading (depot -> current) if adding it
    # keeps the route roughly straight. We approximate this via:
    #   align = (d_cur_feas + d_dep_feas - d_cur_dep)  (small = more collinear/aligned)
    # Actually, by triangle inequality, d_cur_feas + d_dep_feas >= d_cur_dep,
    # and equality holds when the three points are collinear with current between depot and feas.
    # We want nodes where the path depot->current->feas is efficient, i.e.,
    # the "excess" (d_cur_feas + d_dep_feas - d_cur_dep) is small relative to scale.
    # Lower excess = better alignment with the sweep direction.
    excess = d_cur_feas + d_dep_feas - d_cur_dep
    # Normalize excess: lower is better for alignment
    eps = 1e-10
    excess_range = np.max(excess) - np.min(excess) + eps
    align_norm = (excess - np.min(excess)) / excess_range  # 0 = best aligned, 1 = worst
    
    # Compute continuation cost with SOFT-MIN surrogate for smoothness.
    # Instead of hard min(d(cand, depot), min_to_other), use log-sum-exp soft-min.
    # soft_min(a, b) = -logsumexp(-a, -b) approximates min(a,b) smoothly.
    if n_nodes > 1:
        d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
        np.fill_diagonal(d_feas_feas, np.inf)
        min_to_other = np.min(d_feas_feas, axis=1)
        
        # Soft-min of d_dep_feas and min_to_other
        # Use temperature beta for softness: larger beta = sharper (closer to hard min)
        beta = 5.0
        a = -beta * d_dep_feas
        b = -beta * min_to_other
        # logsumexp(a, b) = max(a,b) + log(exp(a-max) + exp(b-max))
        m = np.maximum(a, b)
        logsumexp_val = m + np.log(np.exp(a - m) + np.exp(b - m))
        cont_cost = -logsumexp_val / beta
    else:
        cont_cost = d_dep_feas.copy()
    
    # Normalize terms to have comparable scales
    eps = 1e-10
    savings_norm = (savings - np.min(savings)) / (np.max(savings) - np.min(savings) + eps)
    dist_norm = (d_cur_feas - np.min(d_cur_feas)) / (np.max(d_cur_feas) - np.min(d_cur_feas) + eps)
    cont_norm = (cont_cost - np.min(cont_cost)) / (np.max(cont_cost) - np.min(cont_cost) + eps)
    
    # Weighted score combining savings, distance penalty, alignment, and continuation cost
    # W_sav: weight for savings (higher = prefer high-savings nodes)
    # W_dist: weight for distance penalty (higher = prefer shorter edges)
    # W_align: weight for geometric alignment (higher = prefer sweep-aligned nodes)
    # W_cont: weight for continuation cost (higher = prefer nodes with easy next steps)
    W_sav = 1.0
    W_dist = 0.5
    W_align = 0.4
    W_cont = 0.3
    
    scores = W_sav * savings_norm - W_dist * dist_norm - W_align * align_norm - W_cont * cont_norm
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
