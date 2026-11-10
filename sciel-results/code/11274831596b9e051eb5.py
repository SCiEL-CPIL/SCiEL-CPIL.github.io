"""Combined heuristic: CMDS coords + multi-start NN+2-opt+3-opt reference tour,
with explicit geometric crossing (planarity) repulsion for ACO TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin under 330s


def _mds_coords(dm):
    """Recover 2D coordinates from a distance matrix via classical MDS."""
    n = dm.shape[0]
    D2 = dm ** 2
    row_means = np.mean(D2, axis=1, keepdims=True)
    col_means = np.mean(D2, axis=0, keepdims=True)
    grand_mean = np.mean(D2)
    B = -0.5 * (D2 - row_means - col_means.T + grand_mean)
    B = 0.5 * (B + B.T)  # symmetrize against numerical noise
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    ev = np.maximum(eigenvalues[:2], 0.0)
    coords = eigenvectors[:, :2] * np.sqrt(ev)
    return coords


def _orient(p, q, r):
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if val > 1e-12:
        return 1
    if val < -1e-12:
        return -1
    return 0


def _seg_crosses(a, b, c, d):
    """True if open segments (a,b) and (c,d) properly cross."""
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return o1 != o2 and o3 != o4


def _nn_tour(dist, n, start):
    """Nearest-neighbor tour starting at `start`. Returns tour list."""
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
    return tour


def _tour_len(tour, dist, n):
    return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))


def _two_opt(tour, dist, n, max_passes=40):
    """In-place 2-opt improvement."""
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
                cur = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new < cur - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _three_opt(tour, dist, n, max_passes=12):
    """In-place 3-opt improvement. Tries all 3 ways to reconnect 3 cut edges."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if (i == 0 and k == n - 1):
                        continue
                    # cut edges: (i-1,i),(j-1,j),(k-1,k) with wrap on i-1
                    a = tour[(i - 1) % n]
                    b = tour[i]
                    c = tour[(j - 1) % n]
                    d = tour[j]
                    e = tour[(k - 1) % n]
                    f = tour[k]
                    cur = dist[a, b] + dist[c, d] + dist[e, f]
                    # Three reconnection options (reverse middle segment between j..k)
                    # Option 1: a-d, b-c, e-f  -> keep, just reverse j..k? handle via swaps
                    # We implement the two non-trivial reconnects that reverse the
                    # segment between j and k.
                    new1 = dist[a, d] + dist[b, c] + dist[e, f]
                    if new1 < cur - 1e-12:
                        tour[j:k + 1] = tour[j:k + 1][::-1]
                        # after reversing j..k, edges (j-1,j) and (k,k+1) swap ends
                        improved = True
                        continue
                    new2 = dist[a, f] + dist[b, c] + dist[e, d]
                    if new2 < cur - 1e-12:
                        tour[j:k + 1] = tour[j:k + 1][::-1]
                        # swap the two outer connections
                        tour[i], tour[k] = tour[k], tour[i]
                        improved = True
                        continue
    return tour


def _reference_tour(dist, n):
    """Multi-start NN + 2-opt + 3-opt. Returns best tour and its length."""
    best_tour = None
    best_len = np.inf
    for s in range(n):
        if time.time() > _DEADLINE - 30:
            break
        tour = _nn_tour(dist, n, s)
        tour = _two_opt(tour, dist, n, max_passes=25)
        tour = _three_opt(tour, dist, n, max_passes=8)
        length = _tour_len(tour, dist, n)
        if length < best_len:
            best_len = length
            best_tour = tour[:]
    # Extra polish on the best tour found
    if best_tour is not None and time.time() < _DEADLINE - 20:
        best_tour = _two_opt(best_tour, dist, n, max_passes=40)
        best_tour = _three_opt(best_tour, dist, n, max_passes=15)
        best_len = _tour_len(best_tour, dist, n)
    return best_tour, best_len


def heuristics(distance_matrix):
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]

    # Recover geometry so we can measure real crossings
    coords = _mds_coords(dist)

    # High-quality reference tour (multi-start NN + 2-opt + 3-opt)
    ref_tour, _ = _reference_tour(dist, n)

    # Base: strong inverse-distance prior (5th power) to suppress long edges
    base = 1.0 / (dist + 1e-9) ** 5
    base = np.where(np.eye(n, dtype=bool), 0.0, base)  # zero diagonal

    # Reference-tour edge segments (as coordinate pairs)
    ref_segs = []
    for i in range(n):
        a = ref_tour[i]
        b = ref_tour[(i + 1) % n]
        ref_segs.append((coords[a], coords[b]))

    # Explicit planarity repulsion: count how many reference segments each
    # candidate edge (i,j) crosses. More crossings => less attractive.
    H = base.copy()
    for i in range(n):
        for j in range(i + 1, n):
            seg = (coords[i], coords[j])
            crossings = 0
            for (a, b) in ref_segs:
                if _seg_crosses(seg[0], seg[1], a, b):
                    crossings += 1
            # repulsive factor: divide by (1 + crossings)
            repel = 1.0 / (1.0 + crossings)
            H[i, j] *= repel
            H[j, i] *= repel

    # Strong boost on reference-tour edges (the near-optimal non-crossing skeleton)
    for i in range(n):
        a = ref_tour[i]
        b = ref_tour[(i + 1) % n]
        H[a, b] *= 12.0
        H[b, a] *= 12.0

    # Mild penalty on very long edges (top 20% by distance)
    mask = ~np.eye(n, dtype=bool)
    d_off = dist[mask]
    threshold = np.percentile(d_off, 80)
    H[dist > threshold] *= 0.15

    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    # Normalize to keep magnitudes reasonable
    H = H / H.max() * 1.0

    return H
# EVOLVE-BLOCK-END
