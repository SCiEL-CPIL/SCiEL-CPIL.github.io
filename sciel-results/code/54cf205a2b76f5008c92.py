import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Distance check
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Collinearity check
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _count_convex7_fast(points):
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    eps = 1e-9

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # turn_mat[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # We only need a < b < c
        # dp[l][s][e] = number of convex chains of length l in q,
        # starting at s, ending at e, such that p->q[s]->...->q[e] is convex.
        
        # Precompute turn for pivot p to any pair (s, e) in q
        # turn_p[s, e] = (q[s]-p) x (q[e]-q[s])
        # This is the turn at q[s] for the chain p -> q[s] -> q[e]
        
        # Precompute turn for any triple (a, b, c) in q with a < b < c
        # turn_q[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        
        # We can vectorize this.
        
        # Initialize dp for length 2 (p -> q[s] -> q[e])
        # dp[2][s][e] = 1 if s < e and turn_p[s, e] > eps
        
        # For l >= 3:
        # dp[l][s][e] = sum_{m < e} dp[l-1][s][m] * I(turn_q[m-1, m, e] > eps) ?
        # No, the turn is at q[m], so it involves the point before m.
        # Let chain be p, q[s], ..., q[m], q[e].
        # The last turn is at q[m]. It is (q[m]-q[prev]) x (q[e]-q[m]).
        # So we need to know q[prev].
        
        # This means the state dp[l][s][m] is not sufficient to compute the transition to e,
        # because the transition depends on the predecessor of m.
        
        # Therefore, we must keep a 3D state: dp[l][s][m] where m is the last point,
        # but we also need to sum over all valid predecessors.
        
        # Actually, the standard DP for Erdos-Szekeres k-gons uses:
        # dp[k][i][j] = number of convex chains of length k ending at i, j.
        # Here, the points are sorted by angle around a pivot.
        
        # Let's use the state:
        # dp[l][s][e] = count of convex chains of l points in q, starting at s, ending at e.
        # But as noted, we need the predecessor.
        
        # Alternative:
        # dp[l][s][e] = sum_{m < e} (dp[l-1][s][m] * I(turn(q[m-1], q[m], q[e]) > eps))
        # This is not possible without knowing m-1.
        
        # Correct DP state:
        # dp[l][s][p_idx] = count of convex chains of length l in q, starting at s, ending at p_idx.
        # To extend to e > p_idx, we need to check the turn at p_idx.
        # The turn at p_idx is (q[p_idx-1], q[p_idx], q[e]).
        # But dp[l][s][p_idx] aggregates over all possible p_idx-1.
        # So we cannot simply multiply by a single turn value.
        
        # We need to store the count for each possible predecessor?
        # No, we can do:
        # dp[l][s][e] = sum_{m < e} ( sum_{p < m} dp[l-1][s][p, m] * I(turn(p, m, e) > eps) )
        # where dp[l-1][s][p, m] is 1 if chain s->...->p->m is valid.
        
        # This requires a 4D array dp[l][s][p][m], which is too large.
        
        # For N=33, let's just use the O(N^4) brute force check for all 7-subsets?
        # Number of 7-subsets of 33 is C(33, 7) = 4,272,048.
        # For each subset, check convexity in O(7 log 7) or O(7^2).
        # Total operations: ~4.2M * 49 = 200M. Might be too slow in Python.
        
        # Let's try to optimize the DP with the 3D state and vectorized sums.
        
        # Actually, the seed program's DP was:
        # dp[l][s][e] = sum_{m < e} dp[l-1][s][m] * I(turn(m, e) > eps)
        # But this is incorrect as the turn depends on the predecessor of m.
        
        # Wait, in the seed program, the turn was computed as:
        # turn_mat[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # And the DP was:
        # dp[l][s][e] = sum_{m < e} dp[l-1][s][m] * I(turn_mat[?, m, e] > eps)
        # The seed used:
        # turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
        # dp_slice = dp_prev[s, s:p_idx, p_idx]
        # mask = turn_slice > eps
        # contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
        # dp_curr[s, p_idx, p_idx+1:] = contrib
        
        # This implies that dp_prev[s, s:p_idx, p_idx] is a vector of counts for each possible predecessor.
        # So dp_prev[s, a, b] is the count of chains starting at s, ending at b, with predecessor a.
        # This is a 3D state: dp[l][s][p_idx] where p_idx is the last point, and the state is indexed by the predecessor?
        # No, the seed's dp_prev is (m, m, m), so dp_prev[s, p_idx, e] is the count of chains of length l-1 starting at s, ending at e, with predecessor p_idx?
        # Actually, the seed's dp_prev[s, p_idx, e] is the count of chains of length l-1 starting at s, ending at e, where the point before e is p_idx.
        
        # This is the correct state!
        # dp[l][s][p_idx] is not enough; we need dp[l][s][p_idx, e] = count of chains of length l starting at s, ending at e, with predecessor p_idx.
        
        # So the state is 3D: dp[l][s][p_idx, e] but stored as dp[l][s][p_idx, e] where p_idx < e.
        # Actually, the seed used dp_prev[s, p_idx, e] where p_idx is the predecessor of e.
        
        # Let's re-implement the seed's DP correctly.
        
        # dp[l][s][p_idx, e] = count of convex chains of length l in q, starting at s, ending at e, with predecessor p_idx.
        # l=2: dp[2][s][s, e] = 1 if s < e and turn(p, q[s], q[e]) > eps.
        # l>2: dp[l][s][p_idx, e] = sum_{m < p_idx} dp[l-1][s][m, p_idx] * I(turn(q[m], q[p_idx], q[e]) > eps)
        # Wait, the turn at p_idx is (q[p_idx]-q[m]) x (q[e]-q[p_idx]).
        # So we need to sum over all m < p_idx.
        
        # This is what the seed did.
        
        # Let's implement this.
        
        # dp_prev[s, p_idx, e] = count of chains of length l-1 starting at s, ending at e, with predecessor p_idx.
        # We want dp_curr[s, p_idx, e] = count of chains of length l starting at s, ending at e, with predecessor p_idx.
        
        # For l=2:
        # dp_prev[s, s, e] = 1 if s < e and turn(p, q[s], q[e]) > eps.
        
        # For l>2:
        # dp_curr[s, p_idx, e] = sum_{m < p_idx} dp_prev[s, m, p_idx] * I(turn(q[m], q[p_idx], q[e]) > eps)
        
        # We can vectorize this.
        
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)
        
        # l=2
        for s in range(m):
            for e in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[e, 0] - q[s, 0]
                v2y = q[e, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > eps:
                    dp_prev[s, s, e] = 1
        
        # l=3 to 6
        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # dp_curr[s, p_idx, e] = sum_{m < p_idx} dp_prev[s, m, p_idx] * I(turn(q[m], q[p_idx], q[e]) > eps)
                    # for e > p_idx
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                    turn_slice = np.zeros((p_idx - s, m - p_idx - 1))
                    for m_idx in range(s, p_idx):
                        v1x = q[p_idx, 0] - q[m_idx, 0]
                        v1y = q[p_idx, 1] - q[m_idx, 1]
                        v2x = q[p_idx+1:, 0] - q[p_idx, 0]
                        v2y = q[p_idx+1:, 1] - q[p_idx, 1]
                        cross_vals = v1x * v2y - v1y * v2x
                        turn_slice[m_idx - s, :] = cross_vals
                    mask = turn_slice > eps
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev
        
        # Now, for each chain of length 6 in q, starting at s, ending at e, with predecessor p_idx,
        # we need to check if the closing turns are convex.
        # The closing turns are:
        # 1. turn(q[e], p, q[s]) > eps
        # 2. turn(p, q[s], q[p_idx]) > eps ? No, the chain is p -> q[s] -> ... -> q[p_idx] -> q[e] -> p.
        # The turns are:
        # p -> q[s] -> ... -> q[e] -> p
        # The last two turns are:
        # turn(q[e], p, q[s]) and turn(p, q[s], q[p_idx])?
        # Actually, the turns are:
        # 1. turn(p, q[s], q[i1])
        # 2. turn(q[s], q[i1], q[i2])
        # ...
        # 6. turn(q[i5], q[e], p)
        # 7. turn(q[e], p, q[s])
        
        # So we need to check:
        # turn(q[e], p, q[s]) > eps
        # and turn(p, q[s], q[i1]) > eps (already checked in l=2)
        
        # Wait, the seed checked:
        # v1x = q[c_idx, 0] - q[p_idx, 0]
        # v1y = q[c_idx, 1] - q[p_idx, 1]
        # v2x = p[0] - q[c_idx, 0]
        # v2y = p[1] - q[c_idx, 1]
        # cross1 = v1x * v2y - v1y * v2x
        # if cross1 <= eps: continue
        # v1x = p[0] - q[c_idx, 0]
        # v1y = p[1] - q[c_idx, 1]
        # v2x = q[s, 0] - p[0]
        # v2y = q[s, 1] - p[1]
        # cross2 = v1x * v2y - v1y * v2x
        # if cross2 <= eps: continue
        
        # This checks turn(q[p_idx], q[c_idx], p) and turn(q[c_idx], p, q[s]).
        # But the chain ends at q[c_idx], so the last turn is turn(q[p_idx], q[c_idx], p).
        # And the closing turn is turn(q[c_idx], p, q[s]).
        
        # So we need to check these two turns.
        
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= eps:
                        continue
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= eps:
                        continue
                    total += count

    return total

def _make_lattice_grid(n_points, rng):
    """
    Create a set of points on a sparse grid.
    """
    # Use a 6x6 grid, which has 36 points. We need 33.
    # We can use a 6x6 grid and remove 3 points.
    # Or use a 5x7 grid (35 points) and remove 2.
    
    # Let's use a 6x6 grid.
    grid_size = 6
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.1 + 0.8 * i / (grid_size - 1)
            y = 0.1 + 0.8 * j / (grid_size - 1)
            points.append([x, y])
    
    points = np.array(points)
    
    # Remove 3 random points to get 33 points.
    if len(points) > n_points:
        indices_to_remove = rng.choice(len(points), size=len(points) - n_points, replace=False)
        points = np.delete(points, indices_to_remove, axis=0)
    
    # Add small random perturbation to avoid collinearity.
    points += rng.normal(0, 0.001, points.shape)
    points = np.clip(points, 0.01, 0.99)
    
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Initialize with a lattice grid.
    best_pts = _make_lattice_grid(33, rng)
    
    if not _is_valid_fast(best_pts):
        # Perturb until valid
        for _ in range(100):
            best_pts += rng.normal(0, 0.005, (33, 2))
            best_pts = np.clip(best_pts, 0.01, 0.99)
            if _is_valid_fast(best_pts):
                break
    
    if not _is_valid_fast(best_pts):
        # Fallback: random initialization
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
    
    best_c = _count_convex7_fast(best_pts)
    
    # Hill climbing with multiple points per iteration
    while time_left() > 10:
        # Try moving 1-3 random points
        n_move = rng.integers(1, 4)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            new_pts[idx] = best_pts[idx] + rng.normal(0, 0.03, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
    
    return best_pts
