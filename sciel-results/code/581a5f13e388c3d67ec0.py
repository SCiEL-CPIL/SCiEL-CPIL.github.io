"""
Combined TSP Constructive Heuristic: Model-and-Optimize with Multi-Feature Scoring.

Strategy:
1. Detect new instance (when unvisited count is high).
2. Fit parameters (alpha, beta, gamma, delta) via grid search to minimize tour length.
   - alpha: weight for distance from current node (greedy step).
   - beta: weight for distance to destination node (return cost).
   - gamma: weight for min distance to other unvisited nodes (connectivity/2-step).
   - delta: weight for isolation penalty relative to closest node to dest.
3. Use the fitted parameters to score candidates in select_next_node.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_LIMIT = 140.0  # Safety margin before 150s limit

# State for model-and-optimize
_FIT_PARAMS = None
_INSTANCE_ID = 0
_LAST_UNVISITED_COUNT = -1

def _is_deadline_passed():
    return (time.time() - _START_TIME) > _LIMIT

def _build_tour_fast(distance_matrix, start_node, alpha, beta, gamma, delta):
    """
    Build a full TSP tour using the given parameters.
    Returns the total tour length.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = np.arange(1, n)  # Assume start_node is 0 for simplicity in fitting, 
                                  # but in select_next_node, destination_node is the start.
                                  # The fitting function will pass the correct start.
    
    # For fitting, we assume the tour starts at start_node and returns to start_node.
    # The remaining nodes are all others.
    
    while len(remaining) > 0:
        d_cur = distance_matrix[current, remaining]
        
        scores = d_cur.copy()
        
        if beta > 0:
            d_dest = distance_matrix[remaining, start_node]
            scores += beta * d_dest
        
        if gamma > 0 and len(remaining) > 1:
            # Vectorized min distance to other remaining nodes
            # This is O(N^2) per step, so O(N^3) total for one tour.
            # For N=50, this is 125k ops, which is very fast in NumPy.
            sub_dist = distance_matrix[np.ix_(remaining, remaining)]
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 1:
            d_dest_all = distance_matrix[remaining, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = np.argmin(scores)
        next_node = int(remaining[best_idx])
        total_dist += distance_matrix[current, next_node]
        
        # Remove next_node from remaining
        remaining = np.delete(remaining, best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist

def _fit_params(distance_matrix, start_node, time_budget=5.0):
    """
    Fit alpha, beta, gamma, delta by grid search.
    """
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (1.0, 1.0, 0.0, 0.0)
    
    # Grid of parameters
    # alpha is usually 1.0 (base greedy). We vary the others.
    # Let's fix alpha=1.0 and vary beta, gamma, delta.
    # This reduces the search space.
    
    beta_values = [0.0, 0.1, 0.2, 0.4, 0.6]
    gamma_values = [0.0, 0.1, 0.2, 0.4]
    delta_values = [0.0, 0.1, 0.2]
    
    best_alpha = 1.0
    best_beta = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    deadline = elapsed_start + min(time_budget, max(0.5, _LIMIT - (time.time() - _START_TIME) - 5.0))
    
    for beta in beta_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour_fast(distance_matrix, start_node, 1.0, beta, gamma, delta)
                if tour_len < best_len:
                    best_len = tour_len
                    best_beta = beta
                    best_gamma = gamma
                    best_delta = delta
    
    _FIT_PARAMS = (1.0, best_beta, best_gamma, best_delta)
    return _FIT_PARAMS

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    """
    global _FIT_PARAMS, _INSTANCE_ID, _LAST_UNVISITED_COUNT
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    # A new instance typically starts with n-1 unvisited nodes (if start is visited) 
    # or n unvisited nodes (if start is not yet "consumed" in the list, but here 
    # destination_node is the start, and current_node is the start initially).
    # In the contract, current_node is the current position. 
    # If this is the first step, current_node == destination_node usually? 
    # Or current_node is the start city.
    
    n_total = distance_matrix.shape[0]
    
    # Heuristic for new instance: 
    # If we haven't seen a smaller unvisited count recently, and current count is high.
    is_new_instance = False
    if _LAST_UNVISITED_COUNT < 0 or n_unvisited > _LAST_UNVISITED_COUNT + 5:
         # Significant jump in unvisited count suggests new instance
         # Or if it's the very first call
         if _LAST_UNVISITED_COUNT == -1 or n_unvisited >= n_total - 1:
             is_new_instance = True
    
    if is_new_instance:
        _INSTANCE_ID += 1
        # Fit parameters for this new instance
        # Use destination_node as the start for fitting, as the tour must return there.
        _FIT_PARAMS = _fit_params(distance_matrix, destination_node, time_budget=3.0)
    
    _LAST_UNVISITED_COUNT = n_unvisited
    
    if _FIT_PARAMS is None:
        # Fallback: Standard savings heuristic (alpha=1, beta=1, gamma=0, delta=0)
        alpha, beta, gamma, delta = 1.0, 1.0, 0.0, 0.0
    else:
        alpha, beta, gamma, delta = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor for safety
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = alpha * dists_from_current
    
    if beta > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += beta * dists_to_dest
    
    if gamma > 0 and n_unvisited > 1:
        # Vectorized min distance to other unvisited nodes
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 1:
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
