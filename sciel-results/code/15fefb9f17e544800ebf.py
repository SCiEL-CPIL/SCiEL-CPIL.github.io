import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using vectorized dynamic programming.
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

        # Precompute turn matrix: turn[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # We need turn[k, p, c] for k < p < c
        # And turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        
        # Vectorized turn calculation for q
        # turn[a, b, c] = (q[b,0]-q[a,0])*(q[c,1]-q[b,1]) - (q[b,1]-q[a,1])*(q[c,0]-q[b,0])
        
        # Precompute cross products for all triples a, b, c with a < b < c
        # turn_mat[a, b, c]
        # To save memory and time, we can compute on the fly or use a smaller structure.
        # m is at most 32. m^3 is 32768. This is small.
        
        turn_mat = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_mat[a, b, :] = cross_vals

        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            turn_p[s, :] = v1x * v2y - v1y * v2x

        # DP
        # dp[l][s][p][c]
        # l=2: dp[2][s][s][c] = 1 if turn_p[s, c] > 1e-9
        # Note: The chain starts at s. The "previous" point for the first step is s itself?
        # The definition: chain p -> q[s] -> q[c].
        # Turn p -> q[s] -> q[c] is turn_p[s, c].
        # So dp[2][s][s][c] = 1 if turn_p[s, c] > 1e-9.
        
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        for s in range(m):
            # c > s
            valid_c = np.where(turn_p[s, s+1:] > 1e-9)[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # dp_prev[s, k, p_idx] for k in [s, p_idx-1]
                    # We want to sum over k where turn_mat[k, p_idx, c] > 1e-9
                    # dp_curr[s, p_idx, c] = sum_k dp_prev[s, k, p_idx] * (turn_mat[k, p_idx, c] > 1e-9)
                    
                    # Extract vector of dp values for k
                    # k ranges from s to p_idx-1
                    # dp_prev[s, s:p_idx, p_idx] has shape (p_idx-s,)
                    dp_vec = dp_prev[s, s:p_idx, p_idx]
                    if not np.any(dp_vec > 0):
                        continue
                    
                    # Extract turn matrix slice
                    # turn_mat[k, p_idx, c] for k in [s, p_idx-1], c in [p_idx+1, m-1]
                    # turn_mat[s:p_idx, p_idx, p_idx+1:] has shape (p_idx-s, m-p_idx-1)
                    turn_slice = turn_mat[s:p_idx, p_idx, p_idx+1:]
                    
                    # Mask
                    mask = turn_slice > 1e-9
                    
                    # Sum: dp_curr[s, p_idx, c] = sum_k dp_vec[k] * mask[k, c]
                    # This is dp_vec @ mask
                    contrib = dp_vec @ mask
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            
            dp_prev = dp_curr

        # Final counting
        # We have chains of length 6: p -> q[s] -> ... -> q[p_idx] -> q[c_idx]
        # We need to close the polygon: q[c_idx] -> p -> q[s]
        # Check turn(q[p_idx], q[c_idx], p) and turn(q[c_idx], p, q[s])
        
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue
                    
                    # Turn 1: q[p_idx] -> q[c_idx] -> p
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue

                    # Turn 2: q[c_idx] -> p -> q[s]
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

def _generate_initial_points(rng, strategy):
    if strategy == 0:
        # Corner clusters
        pts = []
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for cx, cy in corners:
            for _ in range(8):
                x = cx + rng.normal(0, 0.02)
                y = cy + rng.normal(0, 0.02)
                pts.append((x, y))
        x = rng.uniform(0.4, 0.6)
        y = rng.uniform(0.4, 0.6)
        pts.append((x, y))
        pts = np.array(pts)
        pts = np.clip(pts, 0.01, 0.99)
        return pts
    elif strategy == 1:
        # Grid
        pts = np.array([(i/6.0, j/6.0) for i in range(6) for j in range(6)])
        # Remove 3 points randomly to get 33
        idx = rng.choice(36, 3, replace=False)
        pts = np.delete(pts, idx, axis=0)
        pts += rng.normal(0, 0.01, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        return pts
    else:
        # Random
        return rng.uniform(0.05, 0.95, (33, 2))

def _hill_climb(pts, rng, time_limit):
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        improved = False
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
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
                if best_c == 1:
                    return best_pts, best_c
                step_size *= 0.95
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve > 20:
            step_size = min(0.2, step_size * 1.2)
            no_improve = 0
            
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_ever_pts = None
    best_ever_c = float('inf')
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    strategies = [0, 1, 2]
    
    for strat in strategies:
        if time_left() < 10:
            break
        pts = _generate_initial_points(rng, strat)
        if not _is_valid_fast(pts):
            continue
        c = _count_convex7_fast(pts)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
            if best_ever_c == 1:
                return best_ever_pts
        
        # Hill climb
        if time_left() > 20:
            pts, c = _hill_climb(pts, rng, min(30, time_left() - 10))
            if c < best_ever_c:
                best_ever_c = c
                best_ever_pts = pts.copy()
                if best_ever_c == 1:
                    return best_ever_pts

    # If we didn't find 1, try more random restarts
    while time_left() > 5:
        pts = _generate_initial_points(rng, 0)
        if not _is_valid_fast(pts):
            continue
        pts, c = _hill_climb(pts, rng, min(10, time_left() - 2))
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
            if best_ever_c == 1:
                return best_ever_pts

    return best_ever_pts
