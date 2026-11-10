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
    if n < 7:
        return 0
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

        # Precompute turn(a,b,c) for all a<b<c
        turn_ok = np.zeros((m, m, m), dtype=bool)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_ok[a, b, b+1:] = cross_vals[b+1:] > TOL

        # Precompute turn from pivot p to q[s] to q[c]
        turn_p = np.zeros((m, m), dtype=bool)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            cross_vals = v1x * v2y - v1y * v2x
            turn_p[s, s+1:] = cross_vals[s+1:] > TOL

        # dp[s][p_idx][c_idx] = number of convex chains of current length
        # starting at index s, with previous point at p_idx, ending at c_idx
        dp_prev = np.zeros((m, m, m), dtype=np.int64)

        # Initialize: chains of length 2 (just p -> q[s])
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:])[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        # Extend chains to length 3, 4, 5, 6
        for l in range(3, 7):
            dp_curr = np.zeros((m, m, m), dtype=np.int64)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    dp_slice = dp_prev[s, s:p_idx, p_idx]
                    if not np.any(dp_slice > 0):
                        continue
                    turn_slice = turn_ok[s:p_idx, p_idx, p_idx+1:]
                    contrib = (dp_slice[:, np.newaxis] * turn_slice).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            dp_prev = dp_curr

        # Count complete 7-gons: need to close the chain
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_prev[s, p_idx, c_idx]
                    if count == 0:
                        continue

                    # Check closing turn 1: (q[p_idx], q[c_idx], p)
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= TOL:
                        continue

                    # Check closing turn 2: (q[c_idx], p, q[s])
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
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

def _generate_parabola_cluster(rng, n_pts, x_center, y_center, spread, curvature):
    """Generate points along a parabolic arc with noise."""
    pts = []
    for i in range(n_pts):
        t = i / max(n_pts - 1, 1)
        t = (t - 0.5) * 2  # range [-1, 1]
        x = x_center + spread * t
        y = y_center + curvature * t * t
        x += rng.normal(0, 0.008)
        y += rng.normal(0, 0.008)
        pts.append((x, y))
    return np.array(pts)

def _generate_initial_parabola(rng, seed_offset=0):
    """
    Generate points using multiple parabolic clusters to minimize convex 7-gons.
    Strategy: Points on concave curves (parabolas opening up/down) have the
    property that most 7-subsets are not in convex position.
    """
    rng = np.random.default_rng(seed_offset)
    pts = []
    
    # Cluster 1: bottom-left, parabola opening up
    pts.extend(_generate_parabola_cluster(rng, 12, 0.2, 0.15, 0.15, 0.3))
    # Cluster 2: top-right, parabola opening down
    pts.extend(_generate_parabola_cluster(rng, 11, 0.8, 0.85, 0.15, -0.3))
    # Cluster 3: center, gentle parabola
    pts.extend(_generate_parabola_cluster(rng, 10, 0.5, 0.5, 0.2, 0.15))
    
    pts = np.array(pts[:33])
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_initial_zigzag(rng, seed_offset=0):
    """
    Generate points using a zigzag pattern. Points alternating above and below
    a central line create many non-convex configurations.
    """
    rng = np.random.default_rng(seed_offset)
    pts = []
    
    n_left = 17
    n_right = 16
    
    # Left half: points zigzagging
    for i in range(n_left):
        x = 0.05 + 0.45 * (i / (n_left - 1))
        if i % 2 == 0:
            y = 0.3 + 0.3 * (i / n_left)
        else:
            y = 0.7 - 0.3 * (i / n_left)
        x += rng.normal(0, 0.01)
        y += rng.normal(0, 0.01)
        pts.append((x, y))
    
    # Right half: points zigzagging
    for i in range(n_right):
        x = 0.5 + 0.45 * (i / (n_right - 1))
        if i % 2 == 0:
            y = 0.7 - 0.3 * (i / n_right)
        else:
            y = 0.3 + 0.3 * (i / n_right)
        x += rng.normal(0, 0.01)
        y += rng.normal(0, 0.01)
        pts.append((x, y))
    
    pts = np.array(pts[:33])
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_initial_grid_distorted(rng, seed_offset=0):
    """
    Generate points using a distorted grid. A regular grid has many convex
    polygons, but distorting it can reduce the count.
    """
    rng = np.random.default_rng(seed_offset)
    pts = []
    
    # 6x6 grid with one extra point
    rows, cols = 6, 6
    for i in range(rows):
        for j in range(cols):
            x = 0.1 + 0.8 * (j / (cols - 1))
            y = 0.1 + 0.8 * (i / (rows - 1))
            # Distort: push points toward center or away
            cx, cy = 0.5, 0.5
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 0:
                scale = 1.0 + 0.3 * rng.normal(0, 0.2)
                x = cx + dx * scale
                y = cy + dy * scale
            x += rng.normal(0, 0.015)
            y += rng.normal(0, 0.015)
            pts.append((x, y))
    
    # Add one more point
    x = 0.5 + rng.normal(0, 0.1)
    y = 0.5 + rng.normal(0, 0.1)
    pts.append((x, y))
    
    pts = np.array(pts[:33])
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _hill_climb(pts, rng, time_limit, best_c):
    """
    Perform hill climbing to minimize the number of convex 7-gons.
    """
    best_pts = pts.copy()
    best_c_local = best_c
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 200
    step_size = 0.04
    min_step = 0.002
    max_step = 0.2
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        offset = rng.normal(0, step_size, 2)
        new_pts[i] = best_pts[i] + offset
        new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c_local:
                best_c_local = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size = max(min_step, step_size * 0.93)
            elif c == best_c_local and best_c_local > 1:
                if rng.random() < 0.2:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try multi-point move
        if not improved and no_improve % 4 == 0:
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 1.8, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c_local:
                    best_c_local = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size = max(min_step, step_size * 0.93)
                elif c == best_c_local and best_c_local > 1:
                    if rng.random() < 0.15:
                        best_pts = new_pts
                        no_improve = 0
        
        # Occasionally increase step size
        if no_improve % 20 == 0:
            step_size = min(max_step, step_size * 1.35)
    
    return best_pts, best_c_local

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate multiple initial configurations
    initializers = [
        lambda s: _generate_initial_parabola(rng, s),
        lambda s: _generate_initial_zigzag(rng, s),
        lambda s: _generate_initial_grid_distorted(rng, s),
    ]
    
    best_ever_pts = None
    best_ever_c = float('inf')
    
    # Try each initializer
    for init_idx, init_fn in enumerate(initializers):
        if time_left() < 50:
            break
        
        seed = 42 + init_idx * 1000
        pts = init_fn(seed)
        c = _count_convex7_fast(pts)
        
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        
        print(f"Init {init_idx}: C={c}, best_ever={best_ever_c}")
    
    # Run hill climbing from the best initial configuration
    if best_ever_pts is None:
        best_ever_pts = _generate_initial_parabola(rng, 42)
        best_ever_c = _count_convex7_fast(best_ever_pts)
    
    current_pts = best_ever_pts.copy()
    current_c = best_ever_c
    
    # Multiple rounds of hill climbing with restarts
    n_rounds = 12
    for round_idx in range(n_rounds):
        if time_left() < 40:
            break
        
        # Use a fresh RNG for each round to explore different regions
        round_rng = np.random.default_rng(42 + round_idx * 777)
        
        # If we've been at the same C for a while, try a new random start
        if round_idx > 0 and round_idx % 4 == 0:
            init_fn = initializers[round_idx % len(initializers)]
            seed = 42 + round_idx * 1234
            new_pts = init_fn(seed)
            new_c = _count_convex7_fast(new_pts)
            if new_c < best_ever_c:
                current_pts = new_pts
                current_c = new_c
                best_ever_c = new_c
                best_ever_pts = new_pts.copy()
                print(f"Round {round_idx}: new start C={current_c}")
            else:
                current_pts = best_ever_pts.copy()
                current_c = best_ever_c
        
        chunk_time = min(45, time_left() - 25)
        if chunk_time < 15:
            break
        
        current_pts, current_c = _hill_climb(current_pts, round_rng, chunk_time, current_c)
        
        if current_c < best_ever_c:
            best_ever_c = current_c
            best_ever_pts = current_pts.copy()
            print(f"Round {round_idx}: improved to C={best_ever_c}")
        else:
            print(f"Round {round_idx}: C={current_c}, best_ever={best_ever_c}")
        
        if best_ever_c <= 1:
            break
    
    # Final polish
    if time_left() > 20:
        polish_pts = best_ever_pts.copy()
        polish_c = best_ever_c
        polish_start = time.time()
        polish_step = 0.0015
        polish_rng = np.random.default_rng(999)
        
        max_polish_iters = 3000
        for it in range(max_polish_iters):
            if time.time() - polish_start > time_left() - 10:
                break
            
            if polish_rng.random() < 0.15:
                n_move = polish_rng.integers(2, 4)
                idx = polish_rng.choice(33, size=n_move, replace=False)
                new_pts = polish_pts.copy()
                for i in idx:
                    offset = polish_rng.normal(0, polish_step, 2)
                    new_pts[i] = polish_pts[i] + offset
                    new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            else:
                i = polish_rng.integers(0, 33)
                new_pts = polish_pts.copy()
                offset = polish_rng.normal(0, polish_step, 2)
                new_pts[i] = polish_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < polish_c:
                    polish_c = c
                    polish_pts = new_pts
                    if polish_c < best_ever_c:
                        best_ever_c = polish_c
                        best_ever_pts = polish_pts.copy()
                    polish_step = max(0.0004, polish_step * 0.96)
                elif c == polish_c and polish_c > 1:
                    if polish_rng.random() < 0.08:
                        polish_pts = new_pts
        
        print(f"After polish: C={polish_c}, best_ever={best_ever_c}")
    
    return best_ever_pts
# EVOLVE-BLOCK-END
