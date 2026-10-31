"""Traveling salesman constructive heuristic: Pre-computed tour with 2-Opt + Or-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_start_time = None
_DEADLINE = 140.0


def _fingerprint(distance_matrix):
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[:10, :10].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)


def _tour_length(tour, dm):
    n = len(tour)
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
            if time_limit and time.time() - _start_time > time_limit:
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
            if time_limit and time.time() - _start_time > time_limit:
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

    for start in range(n):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        t = _nn_tour(start, dm)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _two_opt(t, dm, time_limit=_DEADLINE - 5)
        t = _or_opt(t, dm, time_limit=_DEADLINE - 5)
        t = np.array(t)
        tlen = _tour_length(t, dm)
        if tlen < best_len:
            best_len = tlen
            best_tour = t

    if best_tour is None:
        best_tour = np.arange(n)

    return best_tour


def _next_in_tour(current_node, destination_node, unvisited_nodes, dm):
    tour = _get_tour(dm)
    current_pos = np.where(tour == current_node)[0]
    if len(current_pos) == 0:
        dists = dm[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    current_pos = current_pos[0]
    n = len(tour)
    unvisited_set = set(unvisited_nodes.tolist())

    for offset in range(1, n):
        next_pos = (current_pos + offset) % n
        candidate = tour[next_pos]
        if candidate in unvisited_set:
            return int(candidate)

    dists = dm[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])


def _get_tour(dm):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    fid = _fingerprint(dm)
    if fid in _cache:
        return _cache[fid]

    if time.time() - _start_time > _DEADLINE - 15:
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

    if time.time() - _start_time > _DEADLINE:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    return _next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
