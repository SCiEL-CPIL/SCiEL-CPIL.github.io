"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance (detected by unvisited_nodes size dropping),
     fit parameters (alpha, gamma, delta) via a small grid search.
  2. Use a vectorized 2-step lookahead (soft-min surrogate) for efficiency.
  3. Add a connectivity/isolation penalty term.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\\{c}} d(c,u)
             + delta * (d(c,dest) - min_u d(u,dest))  [isolation penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta) best found by fitting
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin(arr, temperature=1.0):
    """Approximate min via log-sum-exp for smoothness."""
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return arr[0]
    # Use a simple min for speed; soft-min adds complexity without clear benefit here
    return np.min(arr)


def _build_tour(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = np.arange(1, n)  # 0 is start
    visited_mask = np.zeros(n, dtype=bool)
    visited_mask[0] = True
    
    while len(remaining) > 0:
        # Vectorized scoring
        d_cur = distance_matrix[current, remaining]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[remaining, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            # For each candidate c, compute min distance from c to other remaining nodes
            # Vectorized: submatrix of remaining x remaining
            sub_dist = distance_matrix[np.ix_(remaining, remaining)]
            # Set diagonal to inf so we don't pick self
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[remaining, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = np.argmin(scores)
        next_node = int(remaining[best_idx])
        total_dist += distance_matrix[current, next_node]
        visited_mask[next_node] = True
        # Remove from remaining
        remaining = np.delete(remaining, best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Grid of parameters
    alpha_values = [0.0, 0.03, 0.07, 0.12, 0.2, 0.35]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.25]
    delta_values = [0.0, 0.05, 0.15]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    deadline = elapsed_start + min(time_budget, max(0.5, _LIMIT - (time.time() - _start_time) - 5.0))
    
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour(distance_matrix, start_node, alpha, gamma, delta)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_gamma = gamma
                    best_delta = delta
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta)
    return (best_alpha, best_gamma, best_delta)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    """
    global _current_instance_id, _FIT_PARAMS, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance: if unvisited count is high (near start of tour),
    # it's likely a new instance. We track by the size of unvisited_nodes
    # being >= n-1 (full or near-full).
    # Better heuristic: if this is the first call for a new instance.
    # We detect a new instance when unvisited_nodes has length >= 49 (for 50-city)
    # or more generally when it's larger than what we've seen recently.
    # Simple approach: if _FIT_PARAMS is None or we're at the very start of a tour.
    
    # Detect new instance by checking if unvisited count is close to total cities
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)  # First or second step of a new tour
    
    if is_new_instance and _FIT_PARAMS is None:
        _instance_counter += 1
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=3.0)
    elif is_new_instance and _instance_counter > 0:
        # New instance, refit
        _instance_counter += 1
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=2.0)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, delta = 0.05, 0.0, 0.0
    else:
        alpha, gamma, delta = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        # Vectorized 2-step lookahead
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        # Isolation penalty
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
