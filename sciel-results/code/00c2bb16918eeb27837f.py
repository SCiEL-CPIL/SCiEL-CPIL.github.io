"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0

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
    
    # Calculate distances
    # d[current, depot] is constant for this step
    d_cur_dep = distance_matrix[current_node, depot]
    
    # d[depot, feas_nodes]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    
    # d[current, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Dynamic regret-minimization framework:
    # Score each candidate by how its edge cost compares to the average edge cost
    # of the remaining feasible set. This steers the route away from outlier-long
    # edges to maintain uniform segment lengths and prevent premature capacity
    # exhaustion on distant nodes.
    
    # Compute the average distance from current node to all feasible nodes
    avg_d_cur = np.mean(d_cur_feas)
    
    # Compute the average distance from depot to all feasible nodes
    avg_d_dep = np.mean(d_dep_feas)
    
    # Regret score: negative of the difference between the candidate's edge cost
    # and the average edge cost. Lower edge costs relative to the average get
    # higher scores (less regret), encouraging selection of nodes that are
    # "cheaper" than typical options.
    regret_score = (avg_d_cur - d_cur_feas) + (avg_d_dep - d_dep_feas)
    
    # Also incorporate a savings component to maintain some of the original
    # heuristic's benefits
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Combine regret minimization with savings
    # Normalize both components to comparable scales
    regret_range = np.max(regret_score) - np.min(regret_score)
    savings_range = np.max(savings) - np.min(savings)
    
    if regret_range > 0:
        regret_norm = regret_score / regret_range
    else:
        regret_norm = np.zeros_like(regret_score)
    
    if savings_range > 0:
        savings_norm = savings / savings_range
    else:
        savings_norm = np.zeros_like(savings)
    
    # Weighted combination: prioritize regret minimization (0.6) over savings (0.4)
    combined_score = 0.6 * regret_norm + 0.4 * savings_norm
    
    # Select node with maximum combined score
    best_idx = np.argmax(combined_score)
    best_node = feas_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
