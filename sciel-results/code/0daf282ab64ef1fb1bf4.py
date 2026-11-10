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

def _make_cup_cap(n_cup, n_cap, rng):
    pts = []
    for i in range(n_cup):
        t = (i + 0.5) / n_cup
        x = 0.05 + 0.9 * t
        y = 0.05 + 0.15 * (t - 0.5) ** 2
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    for i in range(n_cap):
        t = (i + 0.5) / n_cap
        x = 0.05 + 0.9 * t
        y = 0.85 - 0.15 * (t - 0.5) ** 2
        x += rng.normal(0, 0.002)
        y += rng.normal(0, 0.002)
        pts.append((x, y))
    
    return np.array(pts)

def _make_zigzag(n, rng):
    pts = np.zeros((n, 2))
    x_coords = np.linspace(0.05, 0.95, n)
    x_coords += rng.normal(0, 0.01, n)
    x_coords = np.sort(x_coords)
    
    for i in range(n):
        pts[i, 0] = x_coords[i]
        if i % 2 == 0:
            pts[i, 1] = 0.1 + rng.normal(0, 0.01)
        else:
            pts[i, 1] = 0.9 + rng.normal(0, 0.01)
            
    return np.clip(pts, 0.01, 0.99)

def _make_circle(n, rng, radius=0.45, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += rng.normal(0, 0.05, n)
    angles = np.sort(angles)
    
    pts = np.zeros((n, 2))
    for i in range(n):
        r = radius + rng.normal(0, 0.02)
        pts[i, 0] = center[0] + r * np.cos(angles[i])
        pts[i, 1] = center[1] + r * np.sin(angles[i])
        
    return np.clip(pts, 0.01, 0.99)

def _simulated_annealing(pts, rng, time_limit):
    start_time = time.time()
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    
    current_pts = best_pts.copy()
    current_c = best_c
    
    temp = 1.0
    cool_rate = 0.999
    min_temp = 1e-5
    
    while time.time() - start_time < time_limit:
        # Perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = current_pts.copy()
        for idx in indices:
            new_pts[idx] = current_pts[idx] + rng.normal(0, 0.05, 2)
            new_pts[idx] = np.clip(new_pts[idx], 0.01, 0.99)
        
        if not _is_valid_fast(new_pts):
            continue
            
        new_c = _count_convex7_fast(new_pts)
        
        # Acceptance
        delta = new_c - current_c
        if delta < 0 or (temp > min_temp and rng.random() < np.exp(-delta / temp)):
            current_pts = new_pts
            current_c = new_c
            
            if new_c < best_c:
                best_c = new_c
                best_pts = new_pts.copy()
        
        temp *= cool_rate
        if temp < min_temp:
            temp = min_temp

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    best_pts = None
    best_c = float('inf')
    
    candidates = []
    
    # 1. Cup-Cap
    for n_cup, n_cap in [(16, 17), (17, 16)]:
        pts = _make_cup_cap(n_cup, n_cap, rng)
        pts = np.clip(pts, 0.01, 0.99)
        if _is_valid_fast(pts):
            candidates.append(pts)

    # 2. Zigzag
    for _ in range(5):
        pts = _make_zigzag(33, rng)
        if _is_valid_fast(pts):
            candidates.append(pts)

    # 3. Circle
    for _ in range(5):
        pts = _make_circle(33, rng)
        if _is_valid_fast(pts):
            candidates.append(pts)

    # 4. Random
    for _ in range(10):
        pts = rng.uniform(0.05, 0.95, (33, 2))
        if _is_valid_fast(pts):
            candidates.append(pts)

    for pts in candidates:
        if not _is_valid_fast(pts):
            continue
        c = _count_convex7_fast(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = rng.uniform(0.1, 0.9, (33, 2))
        best_c = _count_convex7_fast(best_pts)

    remaining = time_left()
    if remaining > 5:
        hc_time = remaining * 0.9
        best_pts, best_c = _simulated_annealing(best_pts, rng, hc_time)

    return best_pts
