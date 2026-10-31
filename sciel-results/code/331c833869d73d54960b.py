"""Model-and-optimize TSP constructive heuristic with MST-based residual cost.

Strategy:
  1. For each new instance, fit parameters (alpha, beta, gamma) via grid search.
  2. Score each candidate c by:
     d(cur,c) + alpha * MST_weight(unvisited \ {c}) 
                + beta * d(c, dest) 
                + gamma * min_{u in unvisited \ {c}} d(c, u)
  3. The MST term captures the residual connectivity cost of the remaining
     subgraph, encouraging choices that keep the unvisited set well-connected.

Score per candidate c:
  score(c) = d(cur,c) + alpha*MST(unvisited\{c}) + beta*d(c,dest) + gamma*min_next
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, beta, gamma) best found by fitting
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _mst_weight(distance_matrix, nodes):
    """Compute MST weight of the subgraph induced by `nodes` using Prim's algorithm.
    
    Args:
        distance_matrix: full distance matrix
        nodes: array-like of node indices
    
    Returns:
        Total weight of MST over the induced subgraph.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Build sub-matrix
    sub_dm = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = sub_dm[0].copy()
    
    total = 0.0
    for _ in range(1, n):
        # Find closest node not yet in MST
        min_dist[in_mst] = np.inf
        next_idx = int(np.argmin(min_dist))
        total += min_dist[next_idx]
        in_mst[next_idx] = True
        min_dist = np.minimum(min_dist, sub_dm[next_idx])
    
    return total


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, beta=0.0, gamma=0.0):
    """Build a full TSP tour using MST-based scoring. Returns tour length.
    
    score(c) = d(cur,c) + alpha*MST(unvisited\{c}) + beta*d(c,dest) + gamma*min_next
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        n_rem = len(remaining)
        
        if alpha > 0 and n_rem > 2:
            # For each candidate c, compute MST of remaining \ {c}
            # This is expensive, so we only do it for small remaining sets
            # or use an approximation for larger sets
            for i in range(n_rem):
                others = np.delete(rem_arr, i)
                if len(others) > 1:
                    scores[i] += alpha * _mst_weight(distance_matrix, others)
                else:
                    scores[i] += 0.0
        
        if beta > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += beta * d_dest
        
        if gamma > 0 and n_rem > 2:
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, beta, gamma by grid search on the full instance.
    
    Since MST computation is expensive, we limit the grid search to a
    smaller set of candidates and use a time budget.
    """
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Smaller grid due to MST cost
    alpha_values = [0.0, 0.05, 0.1, 0.15]
    beta_values = [0.0, 0.05, 0.1, 0.15]
    gamma_values = [0.0, 0.05, 0.1]
    
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for beta in beta_values:
            if time.time() > deadline:
                break
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour_fast(distance_matrix, start_node, alpha, beta, gamma)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
    
    # Refine around best found
    if time.time() < deadline - 1.0:
        for sa in [max(0, best_alpha - 0.025), best_alpha, min(0.3, best_alpha + 0.025)]:
            for sb in [max(0, best_beta - 0.025), best_beta, min(0.3, best_beta + 0.025)]:
                for sg in [max(0, best_gamma - 0.025), best_gamma, min(0.2, best_gamma + 0.025)]:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, sa, sb, sg)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = sa
                        best_beta = sb
                        best_gamma = sg
    
    _FIT_PARAMS = (best_alpha, best_beta, best_gamma)
    return (best_alpha, best_beta, best_gamma)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    MST-based residual cost estimation for principled connectivity scoring.
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
        # MST is expensive, so use a tighter time budget for fitting
        tb = 2.5 if _instance_counter == 1 else 1.5
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        alpha, beta, gamma = 0.1, 0.1, 0.05
    else:
        alpha, beta, gamma = _FIT_PARAMS
    
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0 and n_unvisited > 2:
        # For each candidate c, compute MST of remaining \ {c}
        # This is O(n^3) total which is fine for n=50
        for i in range(n_unvisited):
            others = np.delete(unvisited_nodes, i)
            if len(others) > 1:
                scores[i] += alpha * _mst_weight(distance_matrix, others)
            else:
                scores[i] += 0.0
    
    if beta > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += beta * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
