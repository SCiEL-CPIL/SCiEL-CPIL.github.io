"""Circle packing in perimeter-4 rectangle: Differentiable Physics-Based Annealing."""
import numpy as np
import time

N = 21
_TIME_LIMIT = 145.0


def _is_valid(x, y, r, w, h, tol=1e-9):
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            if d2 < rs*rs - tol:
                return False
    return True


def _force_relax_unified(x, y, r, w, h, max_iter=2000, max_time=30.0, start_time=None, corner_strength=5.0, damping=0.95):
    """Symplectic integrator with corner biasing forces to pack circles densely."""
    if start_time is None:
        start_time = time.time()
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    dt = 0.002
    min_dt = 1e-6
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Collision forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap = (min_d - d) / d
                    # Repulsive force
                    fx[i] += 50.0 * overlap * dx
                    fy[i] += 50.0 * overlap * dy
                    fx[j] -= 50.0 * overlap * dx
                    fy[j] -= 50.0 * overlap * dy
                    # Slight radius reduction during collision
                    fr[i] -= 0.1 * overlap
                    fr[j] -= 0.1 * overlap
        
        # Wall forces with corner biasing
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                fx[i] += 100.0 * (r[i] - x[i])
            # Right wall
            if x[i] > w - r[i]:
                fx[i] -= 100.0 * (x[i] - (w - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += 100.0 * (r[i] - y[i])
            # Top wall
            if y[i] > h - r[i]:
                fy[i] -= 100.0 * (y[i] - (h - r[i]))
            
            # Corner biasing: attract to nearest corner
            # Calculate distance to each corner
            corners = [(0, 0), (w, 0), (0, h), (w, h)]
            min_dist = float('inf')
            best_corner = None
            for cx, cy in corners:
                d2 = (x[i] - cx)**2 + (y[i] - cy)**2
                if d2 < min_dist:
                    min_dist = d2
                    best_corner = (cx, cy)
            
            if best_corner is not None:
                cx, cy = best_corner
                dx_corner = cx - x[i]
                dy_corner = cy - y[i]
                d_corner = np.sqrt(dx_corner**2 + dy_corner**2) + 1e-10
                # Attractive force toward corner, scaled by proximity
                strength = corner_strength * (1.0 / (1.0 + d_corner))
                fx[i] += strength * dx_corner / d_corner
                fy[i] += strength * dy_corner / d_corner
        
        # Symplectic integration (Verlet-like)
        vx = (vx + dt * fx) * damping
        vy = (vy + dt * fy) * damping
        vr = (vr + dt * fr) * damping * 0.5
        
        x += dt * vx
        y += dt * vy
        r += dt * vr
        
        # Project to valid region
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        r = np.maximum(r, 0.01)
        
        # Adaptive time step
        max_v = max(np.max(np.abs(vx)), np.max(np.abs(vy)), np.max(np.abs(vr)))
        if max_v > 1.0:
            dt = max(min_dt, dt * 0.8)
        elif max_v < 0.01:
            dt = min(0.01, dt * 1.1)
        
        # Check convergence
        if max_v < 1e-10:
            break
    
    return x, y, r


def _polish_result(x, y, r, w, h, max_rounds=50, time_budget=10.0):
    """Iterative expansion polish: expand each circle to max possible radius."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    start = time.time()
    
    for _ in range(max_rounds):
        if time.time() > start + time_budget:
            break
        improved = False
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return x, y, r


def _expansion_relaxation_polish(x, y, r, w, h, max_iter=500, time_budget=8.0):
    """Combined expansion and relaxation for final polish."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    start = time.time()
    
    for it in range(max_iter):
        if time.time() > start + time_budget:
            break
        
        # Try to expand each circle slightly
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i]:
                r[i] = min(r_max, r[i] * 1.01)  # Expand by 1%
        
        # Relax to resolve any new overlaps
        x, y, r = _force_relax_unified(x, y, r, w, h, max_iter=100, max_time=1.0, start_time=start)
    
    return x, y, r


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate max radius for circle i given current positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _is_valid_move(x_new, y_new, r_new, w, h, tol=1e-9):
    """Check if a move is valid."""
    return _is_valid(x_new, y_new, r_new, w, h, tol)


def _constraint_refine(x, y, r, w, h, max_iter=100, time_budget=5.0):
    """Refine positions to satisfy constraints more tightly."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() > start + time_budget:
            break
        changed = False
        for i in range(N):
            # Check if circle can be moved closer to a wall
            for wall in ['left', 'right', 'bottom', 'top']:
                if wall == 'left' and x[i] > r[i] + 1e-6:
                    x_new = x.copy()
                    x_new[i] = r[i]
                    if _is_valid(x_new, y, r, w, h):
                        x[i] = r[i]
                        changed = True
                elif wall == 'right' and x[i] < w - r[i] - 1e-6:
                    x_new = x.copy()
                    x_new[i] = w - r[i]
                    if _is_valid(x_new, y, r, w, h):
                        x[i] = w - r[i]
                        changed = True
                elif wall == 'bottom' and y[i] > r[i] + 1e-6:
                    y_new = y.copy()
                    y_new[i] = r[i]
                    if _is_valid(x, y_new, r, w, h):
                        y[i] = r[i]
                        changed = True
                elif wall == 'top' and y[i] < h - r[i] - 1e-6:
                    y_new = y.copy()
                    y_new[i] = h - r[i]
                    if _is_valid(x, y_new, r, w, h):
                        y[i] = h - r[i]
                        changed = True
        if not changed:
            break
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_x = best_y = best_r = best_w = best_h = None

    def evaluate(x, y, r, w, h):
        nonlocal best_value, best_x, best_y, best_r, best_w, best_h
        if not _is_valid(x, y, r, w, h):
            return False
        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h
            return True
        return False

    # Multi-start over aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9]
    rng = np.random.default_rng(12345)

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Grid-based initial placement
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        cw = w0 / cols
        ch = h0 / rows
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw
            y[i] = (row + 0.5) * ch
            r[i] = min(cw, ch) / 2.0 * 0.9

        # Physics-based relaxation
        x, y, r = _force_relax_unified(x, y, r, w0, h0, max_iter=1500, max_time=15.0, start_time=start_time, corner_strength=3.0)
        
        # Polish with expansion
        if best_value > 0 and time.time() < deadline - 15:
            x_e, y_e, r_e = _polish_result(x, y, r, w0, h0, max_rounds=30, time_budget=5.0)
            evaluate(x_e, y_e, r_e, w0, h0)
        
        # Further expansion-relaxation polish
        if best_value > 0 and time.time() < deadline - 10:
            x_f, y_f, r_f = _expansion_relaxation_polish(x_e, y_e, r_e, w0, h0, max_iter=200, time_budget=5.0)
            evaluate(x_f, y_f, r_f, w0, h0)
        
        # Constraint refinement
        if best_value > 0 and time.time() < deadline - 8:
            x_c, y_c, r_c = _constraint_refine(x_f, y_f, r_f, w0, h0, max_iter=50, time_budget=3.0)
            evaluate(x_c, y_c, r_c, w0, h0)

    # Final polish on best result
    if best_value > 0 and best_x is not None and time.time() < deadline - 10:
        x_p, y_p, r_p = _polish_result(best_x, best_y, best_r, best_w, best_h, max_rounds=40, time_budget=5.0)
        evaluate(x_p, y_p, r_p, best_w, best_h)
        
        x_er, y_er, r_er = _expansion_relaxation_polish(x_p, y_p, r_p, best_w, best_h, max_iter=300, time_budget=5.0)
        evaluate(x_er, y_er, r_er, best_w, best_h)
        
        x_cr, y_cr, r_cr = _constraint_refine(x_er, y_er, r_er, best_w, best_h, max_iter=80, time_budget=3.0)
        evaluate(x_cr, y_cr, r_cr, best_w, best_h)

    # Fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r_out = best_r.copy()
    for _ in range(10):
        r_trial = r_out * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h, tol=1e-12):
            r_out = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, r_out))
