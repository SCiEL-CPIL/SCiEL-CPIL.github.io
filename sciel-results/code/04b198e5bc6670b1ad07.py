"""Circle packing in perimeter-4 rectangle: SOCP-style alternating optimization."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _compute_max_radii(x, y, w, h):
    """Given fixed positions, compute the maximum valid radius for each circle.
    
    For each circle i, r_i <= min(distance to walls, min_j (dist(i,j) - r_j))
    But since radii are coupled, we solve iteratively:
    Start with r_i = min(wall distances), then iteratively reduce r_i to satisfy
    r_i + r_j <= dist(i,j) for all j.
    """
    n = N
    # Initial max from walls
    r_max = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
    r_max = np.maximum(r_max, 1e-6)
    
    # Iterative reduction to satisfy pairwise constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                allowed = d - r_max[j]
                if allowed < r_max[i]:
                    r_max[i] = max(allowed, 0.0)
                    changed = True
        if not changed:
            break
    
    return np.maximum(r_max, 1e-6)


def _optimize_positions_for_radii(r, w, h, max_iter=2000):
    """Given fixed radii, optimize positions using gradient-based approach.
    
    We want to find positions x, y such that:
    - r_i <= x_i <= w - r_i
    - r_i <= y_i <= h - r_i  
    - dist(i,j) >= r_i + r_j for all i != j
    
    Use a force-directed / gradient descent approach.
    """
    n = N
    # Start from current positions or a reasonable grid
    # We'll use a smart initialization based on aspect ratio
    side = int(np.ceil(np.sqrt(n)))
    cols = side
    rows = int(np.ceil(n / side))
    
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) / cols * w
        y[i] = (row + 0.5) / rows * h
    
    # Clamp to valid range
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    dt = 0.01
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        for i in range(n):
            if x[i] < r[i] + 0.01:
                fx[i] += (r[i] - x[i]) * 50
            if x[i] > w - r[i] - 0.01:
                fx[i] -= (x[i] - (w - r[i])) * 50
            if y[i] < r[i] + 0.01:
                fy[i] += (r[i] - y[i]) * 50
            if y[i] > h - r[i] - 0.01:
                fy[i] -= (y[i] - (h - r[i])) * 50
        
        # Pairwise repulsion
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-9:
                        ux = dx / d
                        uy = dy / d
                    else:
                        # Random direction
                        ux = np.cos(np.random.uniform(0, 2 * np.pi))
                        uy = np.sin(np.random.uniform(0, 2 * np.pi))
                    f = overlap * 20.0
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f
        
        if max_overlap < 1e-10:
            break
        
        # Adaptive step
        max_f = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-10)
        if max_f > 0.5:
            dt = 0.005
        elif max_f < 0.05:
            dt = 0.02
        else:
            dt = 0.01
        
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    return x, y


def _try_aspect_ratio(w, h, time_limit=5.0):
    """Try to pack circles in a rectangle of width w, height h.
    
    Returns (x, y, r, value) or None if failed.
    """
    start = time.time()
    
    # Iterative process: start with small radii, grow them
    # Phase 1: Find feasible positions with moderate radii
    r = np.full(N, min(w, h) / 8.0)
    x, y = _optimize_positions_for_radii(r, w, h, max_iter=500)
    
    if time.time() - start > time_limit * 0.3:
        return None
    
    # Phase 2: Iteratively grow radii and re-optimize positions
    best_r = r.copy()
    best_x = x.copy()
    best_y = y.copy()
    best_val = float(np.sum(r))
    
    for grow_iter in range(50):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Try to grow all radii slightly
        r_trial = r * 1.005
        # Cap by wall distances
        r_trial = np.minimum(r_trial, np.minimum(x, w - x))
        r_trial = np.minimum(r_trial, np.minimum(y, h - y))
        
        # Check if this is feasible with current positions
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            best_r = r.copy()
            best_x = x.copy()
            best_y = y.copy()
            best_val = float(np.sum(r))
        else:
            # Need to re-optimize positions
            x_new, y_new = _optimize_positions_for_radii(r_trial, w, h, max_iter=300)
            if _is_valid(x_new, y_new, r_trial, w, h):
                r = r_trial
                x = x_new
                y = y_new
                best_r = r.copy()
                best_x = x.copy()
                best_y = y.copy()
                best_val = float(np.sum(r))
            else:
                # Try smaller growth
                r_trial = r * 1.002
                r_trial = np.minimum(r_trial, np.minimum(x, w - x))
                r_trial = np.minimum(r_trial, np.minimum(y, h - y))
                x_new, y_new = _optimize_positions_for_radii(r_trial, w, h, max_iter=200)
                if _is_valid(x_new, y_new, r_trial, w, h):
                    r = r_trial
                    x = x_new
                    y = y_new
                    best_r = r.copy()
                    best_x = x.copy()
                    best_y = y.copy()
                    best_val = float(np.sum(r))
                else:
                    break
    
    # Phase 3: Final polish - try to maximize each radius individually
    for i in range(N):
        if time.time() - start > time_limit - 0.2:
            break
        # Binary search for max radius of circle i
        lo = r[i]
        hi = min(x[i], w - x[i], y[i], h - y[i])
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                hi = min(hi, d - r[j])
        hi = max(hi, lo)
        
        if hi > lo + 1e-6:
            for _ in range(15):
                mid = (lo + hi) / 2
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                best_r = r.copy()
                best_x = x.copy()
                best_y = y.copy()
                best_val = float(np.sum(r))
    
    if best_val > 0:
        return best_x, best_y, best_r, best_val
    return None


def _polish_packing(x, y, r, w, h, time_limit=10.0):
    """Polish a valid packing by trying to increase sum of radii."""
    start = time.time()
    n = N
    
    # Force-directed position optimization to free up space
    dt = 0.002
    for it in range(500):
        if time.time() - start > time_limit * 0.5:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if x[i] < r[i] + 0.005:
                fx[i] += (r[i] - x[i]) * 30
            if x[i] > w - r[i] - 0.005:
                fx[i] -= (x[i] - (w - r[i])) * 30
            if y[i] < r[i] + 0.005:
                fy[i] += (r[i] - y[i]) * 30
            if y[i] > h - r[i] - 0.005:
                fy[i] -= (y[i] - (h - r[i])) * 30
        
        # Pairwise repulsion
        max_ov = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    ov = min_d - d
                    max_ov = max(max_ov, ov)
                    if d > 1e-9:
                        ux = dx / d
                        uy = dy / d
                    else:
                        ux = 1.0
                        uy = 0.0
                    f = ov * 15.0
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f
        
        if max_ov < 1e-9:
            break
        
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    # Try to grow radii
    for _ in range(200):
        if time.time() - start > time_limit:
            break
        r_trial = r * 1.001
        r_trial = np.minimum(r_trial, np.minimum(x, w - x))
        r_trial = np.minimum(r_trial, np.minimum(y, h - y))
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            # Try position adjustment
            x_new, y_new = _optimize_positions_for_radii(r_trial, w, h, max_iter=100)
            if _is_valid(x_new, y_new, r_trial, w, h):
                r = r_trial
                x = x_new
                y = y_new
            else:
                break
    
    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
        
        # Time budget per aspect ratio
        time_per = min(8.0, (deadline - time.time() - 10) / max(1, len(aspect_ratios) - aspect_ratios.index(w0)))
        
        result = _try_aspect_ratio(w0, h0, time_limit=time_per)
        if result is not None:
            x, y, r, val = result
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
    
    # Polish the best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w, h = best_result
        polished = _polish_packing(x, y, r, w, h, time_limit=min(15.0, deadline - time.time() - 5))
        if polished is not None:
            x, y, r = polished
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Additional polish rounds
    if best_result is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            x, y, r, w, h = best_result
            polished = _polish_packing(x, y, r, w, h, time_limit=min(5.0, deadline - time.time() - 2))
            if polished is not None:
                x, y, r = polished
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w, h = best_result
    
    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
