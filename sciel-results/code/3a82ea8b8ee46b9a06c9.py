"""Direct heuristic: build a high-quality reference tour, then strongly promote
its edges in the attractiveness matrix so the colony follows a good skeleton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=200):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
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

    # Build a strong reference tour via randomized multi-start NN + 2-opt
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(0)

    def finish(tour):
        nonlocal best_tour, best_len
        t = two_opt(tour)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = t
        return t

    # Nearest-neighbor from node 0
    finish(nearest_neighbor(0))
    # Randomized starts
    num_starts = 60
    for _ in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        s = int(rng.integers(n))
        finish(nearest_neighbor(s))

    # Cheap random-restart 2-opt refinement of the best tour
    for _ in range(10):
        if time.time() > deadline - 5.0:
            break
        t = list(best_tour)
        for _ in range(3):
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        t = two_opt(t)
        finish(t)

    if best_tour is None:
        best_tour = nearest_neighbor(0)
        best_tour = two_opt(best_tour)

    # Build the attractiveness matrix: inverse distance, boosted on reference-tour edges.
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    base = 1.0 / safe

    # Promote the reference tour's directed edges with a large factor.
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] = 1.0e4

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
# EVOLVE-BLOCK-END
