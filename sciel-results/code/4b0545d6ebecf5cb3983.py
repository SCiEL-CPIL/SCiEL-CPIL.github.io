# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    Points are sorted lexicographically, for each pivot the later points are
    sorted by atan2 angle around it, and a turn p -> q -> r counts as convex
    only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    TOL = 1e-9

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

        # Precompute turn matrices efficiently
        # turn_ab[a, b] = q[b] - q[a] as vector
        # We'll compute cross products on the fly for memory efficiency

        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1 = q[s] - p
            v2 = q - q[s]
            turn_p[s, :] = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]

        # dp[s, p_idx, c_idx] = number of convex chains
        # Initialize: chains of length 2 (just p -> q[s] -> q[c])
        dp = np.zeros((m, m, m), dtype=np.int64)
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:] > TOL)[0]
            if len(valid_c) > 0:
                dp[s, s, s+1:][valid_c] = 1

        # Extend chains to length 3, 4, 5, 6
        for l in range(3, 7):
            new_dp = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    dp_slice = dp[s, s:p_idx, p_idx]
                    if not np.any(dp_slice > 0):
                        continue
                    # For each c_idx > p_idx, sum over k in [s, p_idx-1]
                    # where dp[s, k, p_idx] > 0 and turn(q[k], q[p_idx], q[c_idx]) > TOL
                    for c_idx in range(p_idx + 1, m):
                        # Compute turn for all k in [s, p_idx-1]
                        v1 = q[p_idx] - q[s:p_idx]
                        v2 = q[c_idx] - q[p_idx]
                        cross_vals = v1[:, 0] * v2[1] - v1[:, 1] * v2[0]
                        mask = cross_vals > TOL
                        new_dp[s, p_idx, c_idx] = np.sum(dp_slice * mask)
            dp = new_dp

        # Sum up results for length 6 chains and check closing turns
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing turn 1: (q[p_idx], q[c_idx], p)
                    v1 = q[c_idx] - q[p_idx]
                    v2 = p - q[c_idx]
                    cross1 = v1[0] * v2[1] - v1[1] * v2[0]
                    if cross1 <= TOL:
                        continue

                    # Check closing turn 2: (q[c_idx], p, q[s])
                    v1 = p - q[c_idx]
                    v2 = q[s] - p
                    cross2 = v1[0] * v2[1] - v1[1] * v2[0]
                    if cross2 <= TOL:
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

def _generate_initial_points(rng):
    """
    Generate a set of points designed to minimize convex 7-gons.
    Strategy: Place points in a thin zig-zag strip.
    Points are arranged along a diagonal line, alternating slightly above and below.
    This minimizes the number of points in convex position because the points
    are nearly collinear.
    """
    pts = np.zeros((33, 2))
    # Define a thin strip along the diagonal y=x
    # x from 0.1 to 0.9
    x_coords = np.linspace(0.1, 0.9, 33)
    
    # Alternate y-offsets to create a zig-zag
    # The offset should be small to keep points close to the line
    # but large enough to satisfy non-collinearity (area > 1e-9)
    # Area ~ 0.5 * base * height. Base ~ 0.8/33 ~ 0.024.
    # Height needs to be such that 0.5 * 0.024 * h > 1e-9 -> h > 1e-7.
    # We use h ~ 0.01 to be safe and robust.
    
    for i in range(33):
        x = x_coords[i]
        if i % 2 == 0:
            y = x + 0.01
        else:
            y = x - 0.01
        
        # Add small noise to break symmetry and ensure generic position
        x += rng.normal(0, 0.001)
        y += rng.normal(0, 0.001)
        
        pts[i] = [x, y]
        
    pts = np.clip(pts, 0.005, 0.995)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 300:
        # Perturb points slightly
        pts += rng.normal(0, 0.001, (33, 2))
        pts = np.clip(pts, 0.005, 0.995)
        attempts += 1
    
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Perform hill climbing to minimize the number of convex 7-gons.
    Uses adaptive step sizes and multiple-point moves to escape local minima.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 300
    step_size = 0.06
    min_step = 0.005
    max_step = 0.3
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        # Random new position within a neighborhood
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                # Reduce step size on improvement
                step_size = max(min_step, step_size * 0.9)
            elif c == best_c and best_c > 1:
                # Accept neutral moves to explore
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try a larger jump with multiple points
        if not improved and no_improve % 3 == 0:
            n_move = rng.integers(2, 6)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 2.0, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size = max(min_step, step_size * 0.9)
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.25:
                        best_pts = new_pts
                        no_improve = 0
        
        # Occasionally increase step size to escape local minima
        if no_improve % 20 == 0:
            step_size = min(max_step, step_size * 1.5)
        
        # Fine-tuning phase: if step size is very small, do precise adjustments
        if step_size < 0.02 and no_improve > 50:
            # Try very fine adjustments
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, 0.003, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                elif c == best_c and best_c > 1:
                    if rng.random() < 0.2:
                        best_pts = new_pts
                        no_improve = 0
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial points using a structured approach
    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    
    # Perform hill climbing with multiple restarts
    best_ever_pts = best_pts.copy()
    best_ever_c = best_c
    
    restart_count = 0
    while time_left() > 40:
        # Run hill climbing for a chunk of time
        chunk_time = min(40, time_left() - 40)
        if chunk_time < 10:
            break
        best_pts, best_c = _hill_climb(best_pts, rng, chunk_time)
        
        if best_c < best_ever_c:
            best_ever_c = best_c
            best_ever_pts = best_pts.copy()
        
        # If we've found C=1, we can stop early
        if best_ever_c <= 1:
            break
        
        # Restart with a fresh initial configuration
        restart_count += 1
        if time_left() > 60:
            new_pts = _generate_initial_points(rng)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                best_pts = new_pts
                best_c = new_c
                best_ever_pts = new_pts.copy()
                best_ever_c = new_c
            else:
                # Continue from current best
                best_pts = best_ever_pts.copy()
                best_c = best_ever_c
    
    # Final polish: very fine-tuning with small step sizes
    if time_left() > 20:
        polish_pts = best_ever_pts.copy()
        polish_c = best_ever_c
        polish_start = time.time()
        polish_step = 0.0015
        
        while time.time() - polish_start < time_left() - 12:
            i = rng.integers(0, 33)
            new_pts = polish_pts.copy()
            offset = rng.normal(0, polish_step, 2)
            new_pts[i] = polish_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < polish_c:
                    polish_c = c
                    polish_pts = new_pts
                    polish_step = max(0.0005, polish_step * 0.95)
                    if polish_c < best_ever_c:
                        best_ever_c = polish_c
                        best_ever_pts = polish_pts.copy()
                elif c == polish_c:
                    if rng.random() < 0.2:
                        polish_pts = new_pts
        
        if polish_c < best_ever_c:
            best_ever_c = polish_c
            best_ever_pts = polish_pts.copy()
    
    return best_ever_pts
# EVOLVE-BLOCK-END
