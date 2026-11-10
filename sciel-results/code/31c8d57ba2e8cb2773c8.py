import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

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

        # Precompute turn matrix
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # Precompute turn from pivot p
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        
        for s in range(m):
            valid_c = np.where(turn_p[s, s + 1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s + 1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    vec = dp_prev[s, s:p_idx, p_idx]
                    if np.any(vec > 0):
                        mask = (turn_mat[s:p_idx, p_idx, p_idx + 1:] > 1e-9).astype(np.int64)
                        contrib = vec @ mask
                        dp_curr[s, p_idx, p_idx + 1:] = contrib
            dp_prev = dp_curr

        # Sum up for l=6
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
                    if cross1 <= 1e-9:
                        continue

                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue

                    total += count

    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

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

def _generate_concave_curve(rng):
    x_vals = np.linspace(0.05, 0.95, 33)
    y_vals = 0.5 * (1 - (2 * x_vals - 1)**2)
    x_vals += rng.normal(0, 0.005, 33)
    y_vals += rng.normal(0, 0.005, 33)
    pts = np.column_stack((x_vals, y_vals))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _generate_zigzag(rng):
    xs = np.linspace(0.1, 0.9, 33)
    ys = np.zeros(33)
    for i in range(33):
        ys[i] = 0.5 + 0.1 * (1 if i % 2 == 0 else -1)
    
    pts = np.column_stack([xs, ys])
    pts += rng.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _generate_grid(rng):
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 6)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    idx_to_remove = rng.choice(36, size=3, replace=False)
    mask = np.ones(36, dtype=bool)
    mask[idx_to_remove] = False
    pts = pts[mask]
    
    pts += rng.normal(0, 0.01, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _generate_clusters(rng):
    pts = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        for _ in range(8):
            x = cx + rng.normal(0, 0.03)
            y = cy + rng.normal(0, 0.03)
            pts.append((x, y))
    x = 0.5 + rng.normal(0, 0.02)
    y = 0.5 + rng.normal(0, 0.02)
    pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _hill_climb(pts, rng, time_limit):
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 200
    step_size = 0.08
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        if rng.random() < 0.1:
            new_pts[i] = rng.random(2) * 0.8 + 0.1
        else:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size *= 0.97
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        if not improved and no_improve % 15 == 0:
            n_move = rng.integers(2, 6)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                if rng.random() < 0.2:
                    new_pts[i] = rng.random(2) * 0.8 + 0.1
                else:
                    offset = rng.normal(0, step_size * 1.5, 2)
                    new_pts[i] = best_pts[i] + offset
                    new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size *= 0.97

        if no_improve % 50 == 0:
            step_size = min(0.25, step_size * 1.4)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
    
    generators = [
        _generate_concave_curve,
        _generate_zigzag,
        _generate_grid,
        _generate_clusters,
    ]
    
    best_ever_pts = None
    best_ever_c = float('inf')
    
    for gen in generators:
        if time_left() < 120:
            break
        
        pts = gen(rng)
        c = _count_convex7_fast(pts)
        
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
    
    if best_ever_pts is None:
        best_ever_pts = _generate_concave_curve(rng)
        best_ever_c = _count_convex7_fast(best_ever_pts)
    
    while time_left() > 60:
        chunk_time = min(45, time_left() - 60)
        if chunk_time < 10:
            break
        
        best_pts, best_c = _hill_climb(best_ever_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        
        if best_ever_c <= 1:
            break
        
        if time_left() > 90:
            for gen in generators:
                if time_left() < 90:
                    break
                new_pts = gen(rng)
                new_c = _count_convex7_fast(new_pts)
                if new_c < best_ever_c:
                    best_ever_c = new_c
                    best_ever_pts = new_pts.copy()
                    break

    return best_ever_pts
