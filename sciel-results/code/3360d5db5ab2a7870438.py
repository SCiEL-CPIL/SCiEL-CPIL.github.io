"""Optimized reference-tour embedding heuristic:
Focuses computational budget on finding a high-quality reference tour via
extensive multi-start construction and local search (2-opt + OR-opt), then
strongly biases the ACO heuristic towards this tour to ensure the colony
converges on a near-optimal solution."""

import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=500):
        """Iterative 2-opt with early exit on time."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                if i == 0:
                    j_range = range(2, n - 1)
                else:
                    j_range = range(i + 2, n)
                a = t[i]
                b = t[i + 1]
                for j in j_range:
                    c = t[j]
                    d = t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        a = t[i]
                        b = t[i + 1]
                        improved = True
        return t

    def or_opt(t, max_passes=100):
        """Relocate single node to best position."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for k in range(n):
                node = t[k]
                prev = t[(k - 1) % n]
                nxt = t[(k + 1) % n]
                cost_remove = D[prev, node] + D[node, nxt] - D[prev, nxt]
                best_delta = cost_remove
                best_pos = -1
                for pos in range(n - 1):
                    if pos == k or pos == (k - 1) % n:
                        continue
                    a = t[pos]
                    b = t[(pos + 1) % n]
                    if a == prev or b == prev or a == nxt or b == nxt:
                        continue
                    cost_insert = D[a, node] + D[node, b] - D[a, b]
                    delta = cost_remove + cost_insert
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_pos = pos
                if best_pos >= 0:
                    t.pop(k)
                    t.insert(best_pos + 1, node)
                    improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    # Multi-start construction + local search for reference tour
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    def refine(tour):
        nonlocal best_tour, best_len
        # Alternate 2-opt and OR-opt for better convergence
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        t = or_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    # Start from node 0
    refine(nearest_neighbor(0))

    # Randomized starts - increase count significantly
    num_starts = 2000
    for _ in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    # Increase perturbation count
    num_perturbations = 500
    for _ in range(num_perturbations):
        if time.time() > deadline - 10.0:
            break
        t = list(best_tour)
        # Perform 2-3 random 2-opt moves to perturb
        for _ in range(3):
            i = int(rng.integers(n - 1))
            j = int(rng.integers(n - 1))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0))

    # Build heuristic: Base is inverse distance, boosted by reference tour
    base = 1.0 / np.maximum(D, 1e-9)
    base[np.diag_indices(n)] = 0.0
    
    # Boost reference tour edges strongly
    # Using a multiplier instead of absolute value allows some ACO flexibility
    # but keeps the bias very high
    boost_factor = 1000.0
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] *= boost_factor
        base[j, i] *= boost_factor

    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)
    base[np.diag_indices(n)] = 1e-9

    return base
