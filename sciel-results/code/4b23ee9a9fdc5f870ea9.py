"""Circle packing in perimeter-4 rectangle: SLSQP + Local Polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
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

def _generate_initial_positions(w, h, seed):
    """Generate diverse initial positions for circles."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1 * w, 0.9 * w, N)
    y = rng.uniform(0.1 * h, 0.9 * h, N)
    
    # Simple relaxation to reduce initial overlaps
    r_est = 0.15
    for _ in range(50):
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 2 * r_est and d > 1e-6:
                    push = (2 * r_est - d) * 0.5
                    x[i] += push * dx / d
                    y[i] += push * dy / d
                    x[j] -= push * dx / d
                    y[j] -= push * dy / d
        x = np.clip(x, r_est, w - r_est)
        y = np.clip(y, r_est, h - r_est)
        
    r = np.full(N, r_est)
    return x, y, r

def _slsqp_optimize(x0, y0, r0, w, h, max_iter=200):
    """
    Optimize positions and radii using SLSQP.
    Variables: [x_0..x_N, y_0..y_N, r_0..r_N]
    Objective: -sum(r_i)
    """
    z0 = np.concatenate([x0, y0, r0])
    
    def objective(z):
        r = z[2*N:]
        return -np.sum(r)
    
    def jac(z):
        # Gradient of -sum(r) w.r.t z
        g = np.zeros(3*N)
        g[2*N:] = -1.0
        return g
    
    # Constraints: g_c(z) >= 0
    def make_constraint(i, j):
        def fun(z):
            if i == j:
                return np.inf
            xi, yi, ri = z[i], z[N+i], z[2*N+i]
            xj, yj, rj = z[j], z[N+j], z[2*N+j]
            dx = xi - xj
            dy = yi - yj
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-10:
                return -100 # Strong penalty for coincidence
            return d - (ri + rj)
        return fun
    
    def make_wall_constraint(axis, side, i):
        """
        axis: 0 for x, 1 for y
        side: 0 for min, 1 for max
        """
        def fun(z):
            if axis == 0:
                coord = z[i]
                dim = w
            else:
                coord = z[N+i]
                dim = h
            r = z[2*N+i]
            if side == 0:
                return coord - r
            else:
                return dim - coord - r
        return fun

    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 0, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 1, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 0, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 1, i)})
        
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, w - 0.01)) # x
    for i in range(N):
        bounds.append((0.01, h - 0.01)) # y
    for i in range(N):
        bounds.append((0.01, min(w, h)/2)) # r
        
    options = {'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
    
    try:
        res = minimize(objective, z0, method='SLSQP', jac=jac, 
                       bounds=bounds, constraints=constraints, options=options)
        if res.success or res.status == 1:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            return x_opt, y_opt, r_opt
        else:
            return x0, y0, r0
    except Exception:
        return x0, y0, r0

def _polish_result(x, y, r, w, h, iterations=50):
    """
    Local search: try to increase radii by moving circles slightly.
    This helps escape local minima found by SLSQP.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(iterations):
        improved = False
        for i in range(N):
            # Try increasing radius slightly
            delta_r = 0.001
            r_trial = r.copy()
            r_trial[i] += delta_r
            
            # Check if valid
            valid = True
            # Check walls
            if x[i] - r_trial[i] < 0 or x[i] + r_trial[i] > w or y[i] - r_trial[i] < 0 or y[i] + r_trial[i] > h:
                valid = False
            
            # Check neighbors
            if valid:
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_trial[i] + r[j]:
                        valid = False
                        break
            
            if valid:
                r[i] = r_trial[i]
                improved = True
            else:
                # Try moving circle i to make room
                # Simple 4-directional nudge
                for dx_move in [0.01, -0.01]:
                    for dy_move in [0.01, -0.01]:
                        x_trial = x.copy()
                        y_trial = y.copy()
                        x_trial[i] += dx_move
                        y_trial[i] += dy_move
                        
                        # Check if new position is valid with current radii
                        pos_valid = True
                        if x_trial[i] - r[i] < 0 or x_trial[i] + r[i] > w or y_trial[i] - r[i] < 0 or y_trial[i] + r[i] > h:
                            pos_valid = False
                        if pos_valid:
                            for j in range(N):
                                if i == j: continue
                                dx = x_trial[i] - x[j]
                                dy = y_trial[i] - y[j]
                                d = np.sqrt(dx*dx + dy*dy)
                                if d < r[i] + r[j]:
                                    pos_valid = False
                                    break
                                    
                        if pos_valid:
                            # Now try increasing radius at new position
                            r_trial2 = r.copy()
                            # Max radius at new pos
                            r_max = min(x_trial[i], w - x_trial[i], y_trial[i], h - y_trial[i])
                            for j in range(N):
                                if i == j: continue
                                dx = x_trial[i] - x[j]
                                dy = y_trial[i] - y[j]
                                d = np.sqrt(dx*dx + dy*dy)
                                r_max = min(r_max, d - r[j])
                            
                            if r_max > r[i] + 1e-4:
                                x[i] = x_trial[i]
                                y[i] = y_trial[i]
                                r[i] = min(r_max, r[i] + 0.01) # Grow slowly
                                improved = True
                                break
                    if improved:
                        break
        if not improved:
            break
            
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
                
    # Search over aspect ratios
    # w + h = 2 => h = 2 - w
    # w in [0.5, 1.5] covers most useful shapes
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 1.3, 0.9, 1.1, 0.85, 1.15, 0.75, 1.2]
    
    for idx, w in enumerate(aspect_ratios):
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
            
        # Run multiple seeds for this aspect ratio
        # Budget per aspect: roughly (deadline - now - 5) / remaining
        remaining = len(aspect_ratios) - idx
        time_per_aspect = (deadline - time.time() - 5) / remaining
        
        # Try 3 seeds per aspect ratio if time permits
        max_seeds = 3
        seed_time = time_per_aspect / max_seeds
        
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
                
            x, y, r = _generate_initial_positions(w, h, seed * 100 + idx)
            
            # SLSQP optimization
            x_opt, y_opt, r_opt = _slsqp_optimize(x, y, r, w, h)
            
            # Polish
            x_pol, y_pol, r_pol = _polish_result(x_opt, y_opt, r_opt, w, h)
            
            update_best(x_pol, y_pol, r_pol, w, h)
            
            # Also update with unpolished if polish failed or was worse
            update_best(x_opt, y_opt, r_opt, w, h)
            
    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        x, y, r = _generate_initial_positions(w, h, 0)
        best_result = (x, y, r, w, h)
        
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
