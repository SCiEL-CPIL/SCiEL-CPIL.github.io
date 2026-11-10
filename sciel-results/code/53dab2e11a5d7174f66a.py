"""Rewritten program: topological-alignment heuristic for ACO TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # safety margin under 330s


def _segment_crosses(a, b, c, d):
    """Return True if open segments (a,b) and (c,d) properly cross."""
    def orient(p, q, r):
        val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if val > 1e-12:
            return 1
        if val < -1e-12:
            return -1
        return 0

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 != o2 and o3 != o4


def _nn_tour(dist, n, start):
    """Nearest-neighbor tour starting at `start`. Returns (tour, length)."""
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = dist[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    # closed tour length
    length = 0.0
    for i in range(n):
        length += dist[tour[i], tour[(i + 1) % n]]
    return tour, length


def _two_opt(tour, dist, n, max_passes=60):
    """In-place 2-opt improvement. Returns length of best tour."""
    def seg_len(t, i, j):
        return dist[t[i], t[j]]

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # current: a-b ... c-d ; new: a-c ... b-d (reverse segment i+1..j)
                cur = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new < cur - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    length = 0.0
    for i in range(n):
        length += dist[tour[i], tour[(i + 1) % n]]
    return length


def _reference_tour(dist, n):
    """Multi-start NN + 2-opt. Returns best tour and its length."""
    best_tour = None
    best_len = np.inf
    starts = list(range(n))
    # try a subset of starts if n large to control cost
    for s in starts:
        tour, _ = _nn_tour(dist, n, s)
        length = _two_opt(tour, dist, n, max_passes=30)
        if length < best_len:
            best_len = length
            best_tour = tour[:]
    return best_tour, best_len


def heuristics(distance_matrix):
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]

    # Reference tour
    ref_tour, _ = _reference_tour(dist, n)

    # Base: inverse distance with higher power to suppress long edges
    base = 1.0 / (dist + 1e-9) ** 5

    # Build set of reference edges (undirected) and their segments
    ref_edges = set()
    ref_segs = []
    for i in range(n):
        a = ref_tour[i]
        b = ref_tour[(i + 1) % n]
        if a > b:
            a, b = b, a
        ref_edges.add((a, b))
        ref_segs.append((a, b))

    # Compute crossing matrix: for each (i,j), count how many reference segments it crosses
    # Vectorize-ish: for each candidate edge, test against each reference segment
    H = base.copy()

    # Precompute coordinates from distance matrix? We don't have coordinates.
    # We need coordinates to test crossing. Reconstruct via ... we only have distances.
    # Fallback: use a proxy for "crossing" via geometry is not possible without coords.
    # Instead, use a topological proxy: penalize edges that are "long" relative to local structure.
    # Simpler robust approach: reward reference edges, penalize edges that are far.

    # Reward reference edges strongly
    for (a, b) in ref_edges:
        H[a, b] *= 10.0
        H[b, a] *= 10.0

    # Penalize edges that are "crossing-like": use a heuristic based on rank of distance
    # Edges with distance in the top 20% get a penalty factor
    flat = dist.flatten()
    # exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    d_off = dist[mask]
    threshold = np.percentile(d_off, 80)
    H[dist > threshold] *= 0.1

    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    # Normalize to keep magnitudes reasonable
    H = H / H.max() * 1.0

    return H
# EVOLVE-BLOCK-END
