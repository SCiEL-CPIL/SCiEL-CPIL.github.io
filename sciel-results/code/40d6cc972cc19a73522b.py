"""Improved CVRP heuristic combining savings, regret, and 2-step lookahead."""
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
    
    # Calculate Savings
    # Savings = d[current, depot] + d[depot, c] - d[current, c]
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # 2-Step Lookahead Continuation Cost:
    # For each candidate c, estimate the minimum cost to continue the route
    # after visiting c. This is min(d(c, depot), min_{j != c} d(c, j)).
    # This penalizes candidates that are isolated or far from other feasible nodes.
    
    # Distance matrix among feasible nodes
    d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    # For each row i, find min over j != i
    # Set diagonal to inf to exclude self
    np.fill_diagonal(d_feas_feas, np.inf)
    min_to_other = np.min(d_feas_feas, axis=1)
    
    # Continuation cost = min(d(c, depot), min_to_other)
    cont_cost = np.minimum(d_dep_feas, min_to_other)
    
    # Normalize terms to have comparable scales
    # We use a simple min-max normalization for each term
    
    # Savings normalization
    s_min, s_max = np.min(savings), np.max(savings)
    savings_norm = (savings - s_min) / (s_max - s_min + 1e-10)
    
    # Distance normalization
    d_min, d_max = np.min(d_cur_feas), np.max(d_cur_feas)
    dist_norm = (d_cur_feas - d_min) / (d_max - d_min + 1e-10)
    
    # Continuation cost normalization
    c_min, c_max = np.min(cont_cost), np.max(cont_cost)
    cont_norm = (cont_cost - c_min) / (c_max - c_min + 1e-10)
    
    # Weighted Score:
    # We want to maximize savings, minimize distance, and minimize continuation cost.
    # Score = W_s * savings_norm - W_d * dist_norm - W_c * cont_norm
    
    # Tuned weights based on current best (1.0534) and inspirations
    # Current best uses: savings - 0.5 * dist + 1.0 * (mean - dist) ~ savings - 1.5 * dist
    # Inspiration 1 uses: 1.0 * sav - 0.8 * dist - 0.4 * cont
    # Let's try a combination that emphasizes continuation cost slightly more
    # to avoid dead-ends, while keeping savings strong.
    
    W_s = 1.0
    W_d = 0.6
    W_c = 0.5
    
    scores = W_s * savings_norm - W_d * dist_norm - W_c * cont_norm
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
