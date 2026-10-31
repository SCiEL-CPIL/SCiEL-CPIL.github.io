"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta, eta) via a small
     grid search using a fast approximate tour builder.
  2. Use a vectorized 2-step lookahead with soft-min (log-sum-exp) for
     smoothness and better gradient-like behavior.
  3. Add a smooth connectivity/isolation penalty term.
  4. Add a minimax/bottleneck penalty to avoid long edges.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\\{c}} d(c,u)
             + delta * (d(c,dest) - softmin_u d(u,dest))  [isolation penalty]
             + eta * softmax_{u in rem\\{c}} d(c,u)  [bottleneck penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta, eta) best found by fitting
_instance_counter = 0
_soft_temp = 0.5  # Temperature for soft-min operations


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin(arr, temperature=1.0):
    """Approximate min via log-sum-exp for smoothness.
    
    softmin(x) = -T * log(sum(exp(-x_i / T)))
    As T -> 0, approaches min(x). For larger T, smoother.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    if temperature <= 0:
        return float(np.min(arr))
    scaled = -arr / temperature
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return float(np.min(arr))
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(-temperature * log_sum)


def _softmax(arr, temperature=1.0):
    """Approximate max via log-sum-exp for smoothness.
    
    softmax(x) = T * log(sum(exp(x_i / T)))
    As T -> 0, approaches max(x). For larger T, smoother.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    if temperature <= 0:
        return float(np.max(arr))
    scaled = arr / temperature
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return float(np.max(arr))
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(temperature * log_sum)


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0, 
                     eta=0.0, soft_temp=0.5, use_soft=False):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Fast version for fitting: uses simpler operations when use_soft=False.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            if use_soft:
                min_next = np.array([_softmin(sub_dist[i, :], temperature=soft_temp) 
                                     for i in range(len(rem_arr))])
            else:
                min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            d_dest_all = distance_matrix[rem_arr, start_node]
            if use_soft:
                min_d_dest = _softmin(d_dest_all, temperature=soft_temp)
            else:
                min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        if eta > 0 and len(remaining) > 2:
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            if use_soft:
                max_next = np.array([_softmax(sub_dist[i, :], temperature=soft_temp) 
                                     for i in range(len(rem_arr))])
            else:
                max_next = np.max(sub_dist, axis=1)
            scores += eta * max_next
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta, eta by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.0)
    
    alpha_values = [0.0, 0.05, 0.1, 0.15]
    gamma_values = [0.0, 0.05, 0.1]
    delta_values = [0.0, 0.05]
    eta_values = [0.0, 0.05, 0.1]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_eta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                for eta in eta_values:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, delta, eta,
                                                use_soft=False)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = alpha
                        best_gamma = gamma
                        best_delta = delta
                        best_eta = eta
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta, best_eta)
    return (best_alpha, best_gamma, best_delta, best_eta)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    Uses soft-min/soft-max for smoother scoring during tour construction.
    """
    global _current_instance_id, _FIT_PARAMS, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)
    
    if is_new_instance:
        _instance_counter += 1
        tb = 3.5 if _instance_counter == 1 else 2.0
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        alpha, gamma, delta, eta = 0.1, 0.1, 0.05, 0.05
    else:
        alpha, gamma, delta, eta = _FIT_PARAMS
    
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.array([_softmin(sub_dist[i, :], temperature=0.5) 
                         for i in range(n_unvisited)])
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = _softmin(d_dest_all, temperature=0.5)
        scores += delta * (d_dest_all - min_d_dest)
    
    if eta > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        max_next = np.array([_softmax(sub_dist[i, :], temperature=0.5) 
                         for i in range(n_unvisited)])
        scores += eta * max_next
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
