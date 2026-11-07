# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE - (time.time() - _START_TIME)


def _greedy_path_prize(start, prizes, dist, max_len, exclude_start=True):
    """Simulate a greedy path from start: repeatedly pick highest prize
    unvisited node j such that current_len + d[cur,j] + d[j,0] <= max_len.
    Returns total prize collected (including start's prize if not excluded)."""
    n = len(prizes)
    visited = [False] * n
    cur = start
    cur_len = 0.0
    total = 0.0
    if not exclude_start:
        total += prizes[cur]
    visited[cur] = True
    for _ in range(n):
        best_j = -1
        best_prize = -1.0
        for j in range(n):
            if visited[j]:
                continue
            new_len = cur_len + dist[cur, j] + dist[j, 0]
            if new_len <= max_len and prizes[j] > best_prize:
                best_prize = prizes[j]
                best_j = j
        if best_j < 0:
            break
        visited[best_j] = True
        total += prizes[best_j]
        cur_len += dist[cur, best_j]
        cur = best_j
    return total


def heuristics(prizes, distance_matrix, max_len):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Compute for each node j: estimated prize if we visit j as part of a
    # greedy tour starting from 0. This gives a better signal than prize/dist.
    # h[j] = greedy path prize when j is the first move from 0.
    node_value = np.zeros(n)
    for j in range(1, n):
        # Simulate: 0 -> j -> greedy continuation
        d0j = distance_matrix[0, j]
        dj0 = distance_matrix[j, 0]
        if d0j + dj0 > max_len:
            node_value[j] = 0.0
            continue
        # Build a greedy path starting with 0->j
        visited = [False] * n
        visited[0] = True
        visited[j] = True
        cur = j
        cur_len = d0j
        total = prizes[j]
        for _ in range(n):
            best_k = -1
            best_p = -1.0
            for k in range(1, n):
                if visited[k]:
                    continue
                new_len = cur_len + distance_matrix[cur, k] + distance_matrix[k, 0]
                if new_len <= max_len and prizes[k] > best_p:
                    best_p = prizes[k]
                    best_k = k
            if best_k < 0:
                break
            visited[best_k] = True
            total += prizes[best_k]
            cur_len += distance_matrix[cur, best_k]
            cur = best_k
        node_value[j] = total

    # Heuristic: h[i,j] encourages going to j if j has high node_value
    # and the transition i->j is feasible within remaining budget context.
    # Use node_value[j] as the main signal, with a small distance penalty.
    # h[i,j] = node_value[j] / (distance_matrix[i,j] + eps)
    # This makes nearby high-value nodes more attractive.
    eps = 1e-9
    dist_safe = distance_matrix + eps
    h = np.tile(node_value[None, :], (n, 1)) / dist_safe

    # Ensure positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)

    # Safety: check deadline
    if (time.time() - t0) > _remaining_budget() - 5:
        pass  # already computed, just return

    return h
# EVOLVE-BLOCK-END
