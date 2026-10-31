"""Traveling salesman constructive heuristic: Greedy Insertion + 2-Opt + Or-Opt with regret fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_cache = {}
_start_time = None
_DEADLINE = 145.0  # Safety margin before 150s kill


def _fingerprint(distance_matrix):
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[:10, :10].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)


def _tour_length(tour, dm):
    n = len(tour)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total


def _two_opt(tour, dm, time_limit=None):
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if time_limit is not None and time.time() - _start_time > time_limit:
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old = dm[a, b] + dm[c, d]
                new = dm[a, c] + dm[b, d]
                if new < old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(tour, dm, time_limit=None):
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time_limit is not None and time.time() - _start_time > time_limit:
                return tour
            for j in range(1, n - 1):
                if j == i or (i + 1) % n == j:
                    continue
                node = tour[j]
                a = tour[(j - 1) % n]
                b = tour[(j + 1) % n]
                p = tour[i]
                q = tour[(i + 1) % n]
                old = dm[a, node] + dm[node, b] + dm[p, q]
                new = dm[a, b] + dm[p, node] + dm[node, q]
                if new < old - 1e-10:
                    tour.pop(j)
                    ins_idx = i + 1 if j > i else i
                    tour.insert(ins_idx, node)
                    improved = True
    return tour


def _greedy_insertion(dm, start_node):
    """Build a tour using greedy insertion heuristic."""
    n = dm.shape[0]
    # Start with a small initial tour
    # Pick the two closest nodes to start
    best_pair = (0, 1)
    best_dist = dm[0, 1]
    for i in range(n):
        for j in range(i + 1, n):
            d = dm[i, j]
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)
    
    tour = [best_pair[0], best_pair[1]]
    unvisited = set(range(n)) - set(tour)
    
    while unvisited:
        # For each unvisited node, find the best insertion position
        best_cost = float('inf')
        best_node = None
        best_pos = None
        
        for node in unvisited:
            # Try inserting at each position in the tour
            for pos in range(len(tour)):
                prev = tour[pos - 1]
                nxt = tour[pos]
                # Cost of inserting node between prev and nxt
                cost = dm[prev, node] + dm[node, nxt] - dm[prev, nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
                    best_pos = pos
        
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    
    return tour


def _nn_tour(start, dm, unvisited=None):
    n = dm.shape[0]
    if unvisited is None:
        unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        dists = dm[cur, list(unvisited)]
        nxt = list(unvisited)[int(np.argmin(dists))]
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def _compute_best_tour(dm):
    n = dm.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Try greedy insertion from multiple starting pairs
    # We'll try a subset of starting configurations
    starts_to_try = list(range(min(n, 10)))
    
    for start in starts_to_try:
        if time.time() - _start_time > _DEADLINE - 15:
            break
        
        # Greedy insertion tour
        t = _greedy_insertion(dm, start)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 10)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 10)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 10)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 10)
        
        tlen = _tour_length(t, dm)
        if tlen < best_len:
            best_len = tlen
            best_tour = list(t)
    
    # Also try NN from a few starts as backup
    for start in starts_to_try[:5]:
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        t = _nn_tour(start, dm)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 5)
        
        tlen = _tour_length(t, dm)
        if tlen < best_len:
            best_len = tlen
            best_tour = list(t)
    
    if best_tour is None:
        best_tour = list(range(n))
    
    return np.array(best_tour)


def _regret_fallback(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Lightweight regret-based heuristic for fallback."""
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    # Compute regret scores
    # For each candidate, regret = second_min_connection - min_connection
    # where connections include distance from current, to destination, and to other unvisited
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    # Distance from current to each candidate
    dist_from_cur = distance_matrix[current_node, candidates]
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # For regret, consider connections to: current, destination, and other unvisited nodes
    # Simplified: use distance to current + distance to destination + min distance to other unvisited
    
    best_scores = np.zeros(n)
    second_scores = np.zeros(n)
    
    for i in range(n):
        c = candidates[i]
        # Connections: from current, to destination, to each other unvisited
        conns = [dist_from_cur[i], dist_to_dest[i]]
        for j in range(n):
            if j != i:
                conns.append(distance_matrix[c, candidates[j]])
        
        conns.sort()
        best_scores[i] = conns[0]
        second_scores[i] = conns[1] if len(conns) > 1 else conns[0]
    
    regret = second_scores - best_scores
    
    # Combined score: distance from current + regret
    # Weight regret more heavily for nodes that are "hard to place"
    alpha = 0.5  # Regret weight
    
    scores = dist_from_cur + alpha * regret
    
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx])


def _next_in_tour(current_node, destination_node, unvisited_nodes, dm):
    tour = _get_tour(dm)
    current_pos = np.where(tour == current_node)[0]
    if len(current_pos) == 0:
        return _regret_fallback(current_node, destination_node, unvisited_nodes, dm)
    
    current_pos = current_pos[0]
    n = len(tour)
    unvisited_set = set(unvisited_nodes.tolist())
    
    for offset in range(1, n):
        next_pos = (current_pos + offset) % n
        candidate = tour[next_pos]
        if candidate in unvisited_set:
            return int(candidate)
    
    return _regret_fallback(current_node, destination_node, unvisited_nodes, dm)


def _get_tour(dm):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    fid = _fingerprint(dm)
    if fid in _cache:
        return _cache[fid]
    
    if time.time() - _start_time > _DEADLINE - 20:
        return np.arange(dm.shape[0])
    
    tour = _compute_best_tour(dm)
    _cache[fid] = tour
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    if time.time() - _start_time > _DEADLINE:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    return _next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
