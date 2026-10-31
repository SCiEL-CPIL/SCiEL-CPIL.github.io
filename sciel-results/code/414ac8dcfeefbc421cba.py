"""
MST-aware TSP constructive heuristic with per-instance parameter fitting.

Strategy:
1. Detect new instance (when unvisited count is high).
2. Fit parameters (alpha, beta, gamma, delta) via grid search to minimize tour length.
   - alpha: weight for distance from current node (greedy step).
   - beta: weight for MST-based residual cost estimate.
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
_mst_cache = {}

def _is_deadline_passed():
    return (time.time() - _START_TIME) > _LIMIT

def _mst_weight(nodes, distance_matrix):
    """
    Compute the MST weight of the subgraph induced by the given nodes.
    Uses Prim's algorithm. Caches results based on tuple of nodes.
    """
    if len(nodes) <= 1:
        return 0.0
    
    key = tuple(int(x) for x in nodes)
    if key in _mst_cache:
        return _mst_cache[key]
    
    n = len(nodes)
    if n == 2:
        w = distance_matrix[nodes[0], nodes[1]]
        _mst_cache[key] = w
        return w
    
    # Prim's algorithm
    in_mst = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0.0
    total = 0.0
    
    for _ in range(n):
        # Find the node with minimum distance
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break
        
        in_mst[u] = True
        total += best
        
        # Update min distances
        node_u = int(nodes[u])
        for v in range(n):
            if not in_mst[v]:
                node_v = int(nodes[v])
                d = distance_matrix[node_u, node_v]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    _mst_cache[key] = total
    return total

def _build_tour_fast(distance_matrix, start_node, alpha, beta, gamma, delta):
    """
    Build a full TSP tour using the given parameters.
    Returns the total tour length.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = np.array([i for i in range(n) if i != start_node])
    
    # Clear MST cache for this fitting run to avoid stale entries
    _mst_cache.clear()
    
    while len(remaining) > 0:
        if len(remaining) == 1:
            next_node = int(remaining[0])
            total_dist += distance_matrix[current, next_node]
            break
        
        d_cur = distance_matrix[current, remaining]
        
        scores = alpha * d_cur.copy()
        
        if beta > 0:
            # MST-based residual cost estimate
            # Approximate: MST weight of remaining nodes
            # This is expensive, so we use a cheaper approximation for fitting
            # Use min distance to destination as proxy for return cost
            d_dest = distance_matrix[remaining, start_node]
            scores += beta * d_dest
        
        if gamma > 0 and len(remaining) > 1:
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
    
    beta_values = [0.0, 0.2, 0.5, 1.0]
    gamma_values = [0.0, 0.1, 0.3, 0.5]
    delta_values = [0.0, 0.1, 0.3]
    
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
    
    n_total = distance_matrix.shape[0]
    
    # Detect new instance
    is_new_instance = False
    if _LAST_UNVISITED_COUNT < 0 or n_unvisited > _LAST_UNVISITED_COUNT + 5:
        if _LAST_UNVISITED_COUNT == -1 or n_unvisited >= n_total - 1:
            is_new_instance = True
    
    if is_new_instance:
        _INSTANCE_ID += 1
        _mst_cache.clear()
        _FIT_PARAMS = _fit_params(distance_matrix, destination_node, time_budget=3.0)
    
    _LAST_UNVISITED_COUNT = n_unvisited
    
    if _FIT_PARAMS is None:
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
        # Use MST-based residual cost for the actual selection
        # For efficiency, use a cheaper proxy: min distance to destination
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += beta * dists_to_dest
    
    if gamma > 0 and n_unvisited > 1:
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
