"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta) via a small grid search
     using a fast approximate tour builder.
  2. Use a vectorized 2-step lookahead with soft-min for smoothness.
  3. Add a 3-step lookahead term for better connectivity.
  4. Critically: reset per-instance state when a new instance is detected.
  5. Optimize the fitting loop to be very fast to leave time for the actual tour.

Score per candidate c:
  score(c) = d(cur, c) + alpha * d(c, dest) + gamma * softmin_{u in rem\{c}} d(c, u)
            + delta * softmin_{v in rem\{c,u}} d(c, v)
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 118.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None
_instance_counter = 0
_last_dm_checksum = None


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin(arr, temperature=0.1):
    """Soft-min via log-sum-exp for smoothness."""
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return arr[0]
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(arr)
    exp_arr = np.exp(-(arr - min_val) / temperature)
    log_sum = np.log(np.sum(exp_arr))
    return min_val - temperature * log_sum


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0, temp=0.1):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Optimized for speed: 
    - Uses list for remaining nodes.
    - Vectorized scoring for the current step.
    - Soft-min for smoothness.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        n_rem = len(rem_arr)
        
        # Distance from current to all remaining
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if (gamma > 0 or delta > 0) and n_rem > 2:
            # For each candidate c, compute distances to other remaining nodes
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)].copy()
            # Set diagonal to inf so c is not its own nearest neighbor
            np.fill_diagonal(sub_dist, np.inf)
            
            if gamma > 0:
                min_next = np.min(sub_dist, axis=1)
                scores += gamma * min_next
            
            if delta > 0 and n_rem > 3:
                # 3-step lookahead: for each c, find the min distance from c's
                # nearest neighbor to any other remaining node
                # This is more expensive but only used in fitting
                # Approximate: use second min as proxy
                sorted_dists = np.sort(sub_dist, axis=1)
                second_min = sorted_dists[:, 1]
                scores += delta * second_min
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta by grid search on the full instance.
    
    Uses fast tour builder for speed.
    """
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Focused grid for speed
    # Alpha: pull to start. Usually 0 or small.
    alpha_values = [0.0, 0.05, 0.1, 0.15]
    # Gamma: connectivity. Higher is better to avoid traps, but too high might distort local optimality.
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    # Delta: 3-step lookahead
    delta_values = [0.0, 0.05, 0.1]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 8.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    # First pass: fit alpha and gamma with delta=0
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, 0.0)
            if tour_len < best_len:
                best_len = tour_len
                best_alpha = alpha
                best_gamma = gamma
    
    # Second pass: refine delta with best alpha, gamma
    if time.time() < deadline:
        for delta in delta_values:
            if time.time() > deadline:
                break
            tour_len = _build_tour_fast(distance_matrix, start_node, best_alpha, best_gamma, delta)
            if tour_len < best_len:
                best_len = tour_len
                best_delta = delta
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta)
    return (best_alpha, best_gamma, best_delta)


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick check: is this the same matrix as before?
        # Use a small checksum for speed
        # Sample every 5th row/col to be more robust
        step = max(1, n_total // 5)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
    
    return False


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
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_counter += 1
        # Reset parameters for the new instance
        _FIT_PARAMS = None
        # Use more time for first instance, less for subsequent
        # 3 seconds is a good balance for 50 nodes with expanded grid
        tb = 3.0
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, delta = 0.0, 0.2, 0.0
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
    
    if (gamma > 0 or delta > 0) and n_unvisited > 2:
        # Vectorized 2-step lookahead
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        
        if gamma > 0:
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and n_unvisited > 3:
            # 3-step lookahead: use second min as proxy
            sorted_dists = np.sort(sub_dist, axis=1)
            second_min = sorted_dists[:, 1]
            scores += delta * second_min
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
