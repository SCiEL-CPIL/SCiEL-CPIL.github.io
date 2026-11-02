"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check if all circles are inside the rectangle and non-overlapping."""
    if w <= 0 or h <= 0:
        return False
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    for i in range(N):
        if r[i] < 0:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _sum_radii(x, y, r, w, h):
    """Return sum of radii if valid, else 0."""
    if _is_valid(x, y, r, w, h):
        return np.sum(r)
    return 0.0

def _grid_layout(w, h):
    """Create a grid layout of N circles in rectangle w x h."""
    side = int(np.ceil(np.sqrt(N)))
    cols = side
    rows = int(np.ceil(N / cols))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
    r[:] = min(w / cols, h / rows) / 2.0 * 0.95
    return x, y, r

def _random_layout(w, h, seed):
    """Create a random layout of N circles in rectangle w x h."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.05 * w, 0.95 * w, N)
    y = rng.uniform(0.05 * h, 0.95 * h, N)
    r = np.full(N, 0.05)
    return x, y, r

def _push_apart(x, y, r, w, h, n_iter=200, dt=0.01):
    """Simple force-based push-apart to resolve overlaps."""
    for _ in range(n_iter):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    force = 0.5 * overlap / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        # Wall forces
        for i in range(N):
            if x[i] - r[i] < 0:
                fx[i] += 0.5 * (r[i] - x[i])
            if x[i] + r[i] > w:
                fx[i] -= 0.5 * (x[i] + r[i] - w)
            if y[i] - r[i] < 0:
                fy[i] += 0.5 * (r[i] - y[i])
            if y[i] + r[i] > h:
                fy[i] -= 0.5 * (y[i] + r[i] - h)
        # Update positions
        x += dt * fx
        y += dt * fy
        # Clip to rectangle
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    return x, y

def _expand_radii(x, y, r, w, h, n_iter=50):
    """Try to expand radii while maintaining validity."""
    for _ in range(n_iter):
        improved = False
        for i in range(N):
            # Try increasing radius of circle i
            old_r = r[i]
            # Max possible radius given position and walls
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            # Also check against other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    max_r = min(max_r, dist - r[j])
            # Try a small increase
            new_r = min(max_r, old_r * 1.02 + 0.001)
            if new_r > old_r + 1e-8:
                r[i] = new_r
                # Check if still valid
                if _is_valid(x, y, r, w, h):
                    improved = True
                else:
                    r[i] = old_r
        if not improved:
            break
    return r

def _slsqp_optimize(x0, y0, r0, w0, h0):
    """Use SLSQP to optimize the layout."""
    # Parameters: x, y, r, w (h is determined by w: h = 2 - w)
    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return 1e6
        # Penalty for wall violations
        pen = 0.0
        for i in range(N):
            pen += max(0, r[i] - x[i]) ** 2
            pen += max(0, x[i] + r[i] - w) ** 2
            pen += max(0, r[i] - y[i]) ** 2
            pen += max(0, y[i] + r[i] - h) ** 2
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        # We want to maximize sum of radii, so minimize -sum(r) + penalty
        return pen - 0.1 * np.sum(r)
    
    def constraints(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        cons = []
        # Wall constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[0:N][i] - p[2*N:3*N][i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*N] - p[0:N][i] - p[2*N:3*N][i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N:2*N][i] - p[2*N:3*N][i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[3*N]) - p[N:2*N][i] - p[2*N:3*N][i]})
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                             (p[0:N][i] - p[0:N][j])**2 + (p[N:2*N][i] - p[N:2*N][j])**2 - 
                             (p[2*N:3*N][i] + p[2*N:3*N][j])**2})
        # Aspect ratio constraint (keep w between 0.3 and 1.7)
        cons.append({'type': 'ineq', 'fun': lambda p: p[3*N] - 0.3})
        cons.append({'type': 'ineq', 'fun': lambda p: 1.7 - p[3*N]})
        return cons
    
    params0 = np.concatenate([x0, y0, r0, [w0]])
    bounds = []
    for i in range(N):
        bounds.append((0, w0))  # x
    for i in range(N):
        bounds.append((0, 2.0 - w0))  # y
    for i in range(N):
        bounds.append((0.01, 0.5))  # r
    bounds.append((0.3, 1.7))  # w
    
    try:
        res = minimize(objective, params0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
        if res.success or res.fun < objective(params0):
            x = res.x[0:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            w = res.x[3*N]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                return x, y, r, w, h
    except Exception:
        pass
    return x0, y0, r0, w0, h0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_w = 1.0
    best_h = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.4]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2, w/h = aspect => w = aspect*h, aspect*h + h = 2 => h = 2/(aspect+1), w = 2*aspect/(aspect+1)
        w = 2.0 * aspect / (aspect + 1.0)
        h = 2.0 / (aspect + 1.0)
        
        # Try grid layout
        x, y, r = _grid_layout(w, h)
        # Push apart to resolve overlaps
        x, y = _push_apart(x, y, r, w, h, n_iter=100)
        # Expand radii
        r = _expand_radii(x, y, r, w, h, n_iter=30)
        
        if _is_valid(x, y, r, w, h):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                best_w = w
                best_h = h
        
        # Try SLSQP optimization from this starting point
        if time.time() < deadline - 10.0:
            x_opt, y_opt, r_opt, w_opt, h_opt = _slsqp_optimize(x.copy(), y.copy(), r.copy(), w, h)
            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
                    best_r = r_opt.copy()
                    best_w = w_opt
                    best_h = h_opt
        
        # Try a few random restarts
        for seed in range(3):
            if time.time() > deadline - 5.0:
                break
            x_rand, y_rand, r_rand = _random_layout(w, h, seed + int(aspect * 100))
            x_rand, y_rand = _push_apart(x_rand, y_rand, r_rand, w, h, n_iter=150)
            r_rand = _expand_radii(x_rand, y_rand, r_rand, w, h, n_iter=30)
            
            if _is_valid(x_rand, y_rand, r_rand, w, h):
                score = np.sum(r_rand)
                if score > best_score:
                    best_score = score
                    best_x = x_rand.copy()
                    best_y = y_rand.copy()
                    best_r = r_rand.copy()
                    best_w = w
                    best_h = h
    
    # If we have a valid solution, return it
    if best_x is not None and best_score > 0:
        return np.column_stack([best_x, best_y, best_r])
    
    # Fallback: simple grid
    x, y, r = _grid_layout(1.0, 1.0)
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
