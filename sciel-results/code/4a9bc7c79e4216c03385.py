import time
import numpy as np
from itertools import combinations

_DEADLINE = 625.0
_SAFE_MARGIN = 5.0

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

        # turn_mat[a, b, c] = cross product of (q[b]-q[a]) and (q[c]-q[b])
        # We only need a < b < c
        # Precompute turn matrix for q
        # turn[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        
        # We will use a 3D array dp[l][s][e]
        # But as analyzed, we need the predecessor.
        
        # Let's use the method from the seed which seemed to work.
        # It used a 3D array and vectorized the inner loop.
        
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # Initialize dp[2]
        # dp[2][s, s, e] = 1 if turn(p, q[s], q[e]) > eps
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > eps)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # dp_prev[s, s:p_idx, p_idx] contains counts for chains of length l-1 ending at p_idx
                    # We need to extend to c > p_idx if turn(q[prev], q[p_idx], q[c]) > eps
                    # The state dp_prev[s, prev, p_idx] gives the count for chains ending at p_idx with prev as the second-to-last point.
                    # So we sum over prev < p_idx.
                    
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                        
                    # turn_mat[prev, p_idx, c]
                    # dp_prev[s, prev, p_idx]
                    
                    # Vectorized sum over prev
                    # turn_slice: (p_idx - s) x (m - p_idx - 1)
                    # dp_slice: (p_idx - s) x 1
                    # mask: (p_idx - s) x (m - p_idx - 1)
                    
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    mask = turn_slice > eps
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev

        # Final check: closing the loop
        # Chain: p -> q[s] -> ... -> q[c]
        # We need turn(q[c], p, q[s]) > eps AND turn(p, q[s], ...)?
        # Wait, the chain is p -> q[s] -> ... -> q[c].
        # The turns are:
        # 1. turn(p, q[s], q[next]) > eps (handled in dp[2] init)
        # 2. turn(q[prev], q[cur], q[next]) > eps (handled in dp loop)
        # 3. Closing turns:
        #    a. turn(q[c], p, q[s]) > eps ? No, the polygon is p, q[s], ..., q[c].
        #       The edges are (p, q[s]), (q[s], ...), ..., (q[prev], q[c]), (q[c], p).
        #       So the turns are:
        #       - At q[s]: turn(p, q[s], q[next])
        #       - ...
        #       - At q[c]: turn(q[prev], q[c], p)
        #       - At p: turn(q[c], p, q[s])
        
        # So we need:
        # 1. turn(q[prev], q[c], p) > eps. But dp_prev[s, prev, c] doesn't check the turn to p.
        #    It only checks turns within the chain q[s]->...->q[c].
        #    So we must check turn(q[prev], q[c], p) > eps for each valid chain ending at c.
        # 2. turn(q[c], p, q[s]) > eps.
        
        for s in range(m):
            for c_idx in range(s + 1, m):
                # Check turn(q[c], p, q[s])
                v1x = p[0] - q[c_idx, 0]
                v1y = p[1] - q[c_idx, 1]
                v2x = q[s, 0] - p[0]
                v2y = q[s, 1] - p[1]
                cross_close = v1x * v2y - v1y * v2x
                if cross_close <= eps:
                    continue
                
                # Sum over all valid chains ending at c_idx
                # dp_prev[s, prev, c_idx] for prev in s+1 .. c_idx-1
                # For each such prev, check turn(q[prev], q[c_idx], p) > eps
                # If so, add dp_prev[s, prev, c_idx]
                
                # turn_mat[prev, c_idx, ?]
                # We need turn(q[prev], q[c_idx], p).
                # p is not in q. So we compute it.
                
                for prev in range(s + 1, c_idx):
                    count = dp_prev[s, prev, c_idx]
                    if count == 0:
                        continue
                    v1x = q[c_idx, 0] - q[prev, 0]
                    v1y = q[c_idx, 1] - q[prev, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > eps:
                        total += count

    return total

def _make_cup_cap(n_cup, n_cap, rng):
    """
    Create a cup-cap construction.
    Cup: n_cup points on a convex curve (lower part)
    Cap: n_cap points on a concave curve (upper part)
    """
    pts = []
    
    # Cup: points along a convex arc at the bottom
    # Use a parabola y = a*x^2 shifted to be in [0,1]
    for i in range(n_cup):
        t = (i + 0.5) / n_cup  # spread in (0,1)
        x = 0.05 + 0.9 * t
        y = 0.05 + 0.15 * (t - 0.5) ** 2  # shallow parabola
        # Add small random perturbation to avoid collinearity
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    # Cap: points along a concave arc at the top
    for i in range(n_cap):
        t = (i + 0.5) / n_cap
        x = 0.05 + 0.9 * t
        y = 0.85 - 0.15 * (t - 0.5) ** 2  # inverted parabola
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Try multiple cup-cap configurations
    best_pts = None
    best_c = float('inf')
    
    # Try different splits
    splits = [(16, 17), (17, 16), (15, 18), (18, 15), (14, 19), (19, 14)]
    
    for n_cup, n_cap in splits:
        if n_cup + n_cap != 33:
            continue
        
        pts = _make_cup_cap(n_cup, n_cap, rng)
        pts = np.clip(pts, 0.01, 0.99)
        
        if not _is_valid_fast(pts):
            # Perturb until valid
            for _ in range(50):
                pts += rng.normal(0, 0.003, (33, 2))
                pts = np.clip(pts, 0.01, 0.99)
                if _is_valid_fast(pts):
                    break
        
        if not _is_valid_fast(pts):
            continue
        
        c = _count_convex7_fast(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()
    
    if best_pts is None:
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
