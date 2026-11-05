"""Improved CVRP heuristic with dynamic regret-minimization framework."""
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
    
    # Dynamic Regret-Minimization Framework:
    # The score for each candidate is the negative of the difference between the current edge cost 
    # and the average edge cost of the remaining feasible set.
    # This steers the route away from outlier-long edges to maintain uniform segment lengths.
    
    mean_dist = np.mean(d_cur_feas)
    
    # Regret term: penalize edges that are longer than the average
    # We want to minimize (d_cur_feas - mean_dist), so we subtract it from the score
    # Or equivalently, we add - (d_cur_feas - mean_dist) to the score
    # Let's define: score = savings - W * d_cur_feas - Lambda * (d_cur_feas - mean_dist)
    # Since mean_dist is constant for all candidates, it doesn't affect the argmax.
    # So effectively, we are just increasing the penalty on distance.
    # But the directive says: "score for each candidate is the negative of the difference between the current edge cost and the average edge cost"
    # So score_regret = -(d_cur_feas - mean_dist) = mean_dist - d_cur_feas
    # This is just -d_cur_feas (up to a constant).
    # To make it more interesting, maybe we combine it with savings.
    
    # Let's try: score = savings - W * d_cur_feas + Beta * (mean_dist - d_cur_feas)
    # This effectively increases the weight on distance.
    # Let's use a simple combination:
    # score = savings - (W + Beta) * d_cur_feas
    
    # Alternatively, let's stick closer to the directive:
    # "score for each candidate is the negative of the difference between the current edge cost and the average edge cost"
    # score_regret = mean_dist - d_cur_feas
    # This favors shorter edges.
    # Let's combine it with savings:
    # score = savings + score_regret = savings + mean_dist - d_cur_feas
    # Since mean_dist is constant, this is equivalent to savings - d_cur_feas.
    
    # Let's try a weighted combination:
    # score = savings - W * d_cur_feas + R * (mean_dist - d_cur_feas)
    # where R is a regret coefficient.
    
    W = 0.5  # Savings weight
    R = 1.0  # Regret weight
    
    scores = savings - W * d_cur_feas + R * (mean_dist - d_cur_feas)
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    # Optional: If the best node is too far (outlier), consider going to depot?
    # Or just trust the heuristic.
    
    return best_node
# EVOLVE-BLOCK-END
