import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 20.0

def _is_valid_fast_v2(pts):
    """Faster validity check using vectorized operations."""
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Check pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Check collinearity using vectorized approach
    # For each triple (i,j,k), check if area < 1e-9
    n = 33
    # Use the cross product formula: x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # We can compute this for all triples efficiently
    # Actually, let's just do it in a tight loop but with early exit
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
                if abs(area) < 1e-9:
                    return False
    return True

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

def _generate_concave_arc_initial(rng):
    """
    Generate points on a concave arc (parabola-like curve) in the unit square.
    Points on a concave curve have the property that many subsets are non-convex
    because interior points lie inside the convex hull of others.
    """
    # Use a downward-opening parabola: y = a - b*x^2
    # Place 33 points along this curve with some jitter
    n = 33
    # Parameterize along the curve
    t = np.linspace(0.05, 0.95, n)
    # Parabola: y = 0.95 - 1.8*(x - 0.5)^2 * 2 = 0.95 - 3.6*(x-0.5)^2
    # At x=0.5, y=0.95; at x=0.05, y=0.95-3.6*0.2025=0.221; at x=0.95, y=0.221
    x = t
    y = 0.95 - 3.6 * (x - 0.5) ** 2
    
    pts = np.column_stack([x, y])
    
    # Add small random perturbations to break exact collinearity
    jitter = rng.normal(0, 0.015, (n, 2))
    pts += jitter
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.02, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_mixed_initial(rng):
    """
    Generate a mix of points: some on a concave arc, some scattered inside.
    This creates a configuration where many 7-point subsets include interior points.
    """
    n = 33
    n_arc = 20  # Points on the arc
    n_interior = n - n_arc  # Interior points
    
    # Arc points
    t = np.linspace(0.05, 0.95, n_arc)
    x_arc = t
    y_arc = 0.95 - 3.6 * (x_arc - 0.5) ** 2
    arc_pts = np.column_stack([x_arc, y_arc])
    
    # Interior points: cluster in the middle-lower region
    # These should be inside the convex hull of the arc points
    x_int = rng.uniform(0.15, 0.85, n_interior)
    y_int = rng.uniform(0.1, 0.5, n_interior)
    int_pts = np.column_stack([x_int, y_int])
    
    pts = np.vstack([arc_pts, int_pts])
    
    # Add small perturbations
    pts += rng.normal(0, 0.012, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.02, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_zigzag_initial(rng):
    """
    Generate points in a zigzag pattern to create many non-convex configurations.
    """
    n = 33
    # Create a zigzag path
    x = np.linspace(0.05, 0.95, n)
    # Alternate between high and low y values
    y = np.zeros(n)
    for i in range(n):
        if i % 2 == 0:
            y[i] = 0.85 - 0.3 * (i / n)
        else:
            y[i] = 0.15 + 0.3 * (i / n)
    
    pts = np.column_stack([x, y])
    pts += rng.normal(0, 0.015, (n, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 500:
        n_fix = rng.integers(1, 4)
        fix_idx = rng.choice(n, size=n_fix, replace=False)
        pts[fix_idx] += rng.normal(0, 0.02, (n_fix, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _simulated_annealing(pts, rng, time_limit):
    """
    Simulated annealing to minimize convex 7-gons.
    Uses temperature-based acceptance to escape local minima.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    current_pts = best_pts.copy()
    current_c = best_c
    
    start_time = time.time()
    t0 = time.time()
    total_time = time_limit
    T = 0.5  # Initial temperature
    T_min = 0.001
    cooling = 0.997
    
    while time.time() - start_time < time_limit:
        # Generate a move: perturb 1-3 points
        n_move = rng.integers(1, 4)
        idx = rng.choice(33, size=n_move, replace=False)
        new_pts = current_pts.copy()
        step_size = 0.02 + 0.03 * T / 0.5  # Scale step with temperature
        for i in idx:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = current_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if not _is_valid_fast(new_pts):
            T *= cooling
            continue
        
        c_new = _count_convex7_fast(new_pts)
        
        # Accept if better, or with probability exp(-delta/T)
        if c_new < current_c:
            current_pts = new_pts
            current_c = c_new
            if c_new < best_c:
                best_c = c_new
                best_pts = new_pts.copy()
        else:
            delta = c_new - current_c
            if delta > 0:
                accept_prob = np.exp(-delta / max(T, T_min))
                if rng.random() < accept_prob:
                    current_pts = new_pts
                    current_c = c_new
        
        T *= cooling
        if T < T_min:
            T = 0.1  # Reheat occasionally
            if rng.random() < 0.3:
                # Kick: move multiple points
                n_kick = rng.integers(3, 8)
                kick_idx = rng.choice(33, size=n_kick, replace=False)
                current_pts[kick_idx] += rng.normal(0, 0.05, (n_kick, 2))
                current_pts = np.clip(current_pts, 0.01, 0.99)
                if _is_valid_fast(current_pts):
                    current_c = _count_convex7_fast(current_pts)
                else:
                    current_pts = best_pts.copy()
                    current_c = best_c
    
    return best_pts, best_c

def _hill_climb_arc(pts, rng, time_limit):
    """
    Hill climbing with arc-aware moves: try to keep points near concave structures.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 100
    step_size = 0.025
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Single point perturbation
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        elif move_type < 0.7:
            # Move point toward a concave position (lower y for middle x)
            i = rng.integers(0, 33)
            new_pts = best_pts.copy()
            x_i = best_pts[i, 0]
            # Target: lower y value proportional to (x-0.5)^2
            target_y = 0.9 - 3.0 * (x_i - 0.5) ** 2
            target_y = max(0.05, min(0.95, target_y))
            new_pts[i, 1] = best_pts[i, 1] * 0.5 + target_y * 0.5 + rng.normal(0, 0.01)
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        else:
            # Multi-point move
            n_move = rng.integers(2, 5)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
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
            elif c == best_c and best_c > 1:
                if rng.random() < 0.3:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Periodic kick
        if no_improve > 0 and no_improve % 20 == 0:
            n_kick = rng.integers(3, 6)
            kick_idx = rng.choice(33, size=n_kick, replace=False)
            new_pts = best_pts.copy()
            for i in kick_idx:
                new_pts[i] += rng.normal(0, step_size * 2, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
        
        if no_improve % 40 == 0:
            step_size = min(0.1, step_size * 1.2)

    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)
    
    # Try multiple initialization strategies
    init_strategies = [
        _generate_concave_arc_initial,
        _generate_mixed_initial,
        _generate_zigzag_initial,
    ]
    
    best_ever_pts = None
    best_ever_c = float('inf')
    
    # Evaluate all initializations
    for init_fn in init_strategies:
        if time_left() < 30:
            break
        pts = init_fn(rng)
        c = _count_convex7_fast(pts)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        print(f"Init C: {c}, best_ever: {best_ever_c}, t={time.time()-start_time:.1f}")
    
    if best_ever_pts is None:
        # Fallback
        best_ever_pts = _generate_concave_arc_initial(rng)
        best_ever_c = _count_convex7_fast(best_ever_pts)
    
    # Phase 1: Simulated annealing from best init
    if time_left() > 60:
        chunk = min(120, time_left() - 60)
        pts, c = _simulated_annealing(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        print(f"After SA: C={c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
    
    # Phase 2: Hill climbing
    if time_left() > 40:
        chunk = min(90, time_left() - 40)
        pts, c = _hill_climb_arc(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        print(f"After HC: C={c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
    
    # Phase 3: More SA with different seed
    if time_left() > 40:
        rng2 = np.random.default_rng(123)
        chunk = min(60, time_left() - 40)
        pts, c = _simulated_annealing(best_ever_pts, rng2, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        print(f"After SA2: C={c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
    
    # Phase 4: Final fine-tuning
    if time_left() > 20:
        chunk = min(30, time_left() - 20)
        pts, c = _hill_climb_arc(best_ever_pts, rng, chunk)
        if c < best_ever_c:
            best_ever_c = c
            best_ever_pts = pts.copy()
        print(f"Final: C={c}, best_ever={best_ever_c}, t={time.time()-start_time:.1f}")
    
    return best_ever_pts
