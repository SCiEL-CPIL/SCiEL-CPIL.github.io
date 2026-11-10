import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0

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
    """
    Count convex 7-gons using the DP described in the task contract.
    Points are sorted lexicographically. For each pivot i, later points are
    sorted by atan2 angle around it. A turn p -> q -> r counts as convex only if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    """
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

        # dp[l][s][e]
        # l: length of chain in q (1 to 6)
        # s: start index
        # e: end index
        
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
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > eps)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    if not np.any(dp_prev[s, s:p_idx, p_idx] > 0):
                        continue
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    mask = turn_slice > eps
                    contrib = (dp_slice[:, np.newaxis] * mask).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev, dp_curr = dp_curr, dp_prev

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

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Initial points: Zigzag
    pts = np.zeros((33, 2))
    for i in range(33):
        pts[i, 0] = i / 32.0
        pts[i, 1] = 0.5 + 0.4 * (-1)**i * np.sin(i * 0.1) # Wavy zigzag
        # Add small noise to avoid collinearity
        pts[i, 0] += rng.normal(0, 0.001)
        pts[i, 1] += rng.normal(0, 0.001)
    
    best_pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    if not _is_valid_fast(best_pts):
        for _ in range(100):
            best_pts += rng.normal(0, 0.005, (33, 2))
            best_pts = np.clip(best_pts, 0.01, 0.99)
            if _is_valid_fast(best_pts):
                break
                
    best_c = _count_convex7_fast(best_pts)
    
    # Simulated Annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    
    while time_left() > 5:
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        # Move point i
        new_pts[i] = best_pts[i] + rng.normal(0, 0.05, 2)
        new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            delta = c - best_c
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                best_c = c
                best_pts = new_pts
                T *= cooling

    return best_pts
