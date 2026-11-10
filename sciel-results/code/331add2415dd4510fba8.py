"""Improved heuristic for ACO-TSP: k-NN geometry + consensus 2-opt tour edge set."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(dm, tour):
    """In-place 2-opt improvement on a closed tour list (node indices)."""
    n = len(tour)
    if n < 4:
        return tour
    best = sum(dm[tour[i], tour[(i + 1) % n]] for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                # remove (a,b) and (c,d), add (a,c) and (b,d)
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    best += delta
                    improved = True
    return tour


def _nearest_neighbor_tour(dm, start=0):
    """Greedy nearest-neighbor tour from `start`."""
    n = dm.shape[0]
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = dm[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def _build_consensus_edge_set(dm):
    """Return a (0,1) matrix marking edges of a 2-opt-refined NN tour."""
    n = dm.shape[0]
    tour = _nearest_neighbor_tour(dm, 0)
    tour = _two_opt(dm, tour)
    edge = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        edge[a, b] = 1.0
    return edge


def heuristics(distance_matrix):
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin under 300s

    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]

    # k-NN inverse-distance term (power 0.5 to avoid extreme spikes)
    eps = 1e-9
    inv = 1.0 / (dm + eps)
    # soften: use sqrt so far edges aren't over-penalised
    base = np.sqrt(inv)

    # consensus tour edge set (cheap: one NN tour + 2-opt)
    if time.time() < deadline:
        edge = _build_consensus_edge_set(dm)
        # boost edges on the consensus tour; keep base structure elsewhere
        heuristic = base * (1.0 + 5.0 * edge)
    else:
        heuristic = base

    # enforce positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
