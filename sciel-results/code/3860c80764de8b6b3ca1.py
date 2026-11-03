"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill

def _perimeter_constraint(x, y, r):
    """Return (width, height) of bounding rectangle."""
    xmin = np.min(x - r)
    xmax = np.max(x + r)
    ymin = np.min(y - r)
    ymax = np.max(y + r)
    return xmax - xmin, ymax - ymin

def _check_validity(x, y, r, tol=1e-6):
    """Check if configuration is valid: disjoint interiors, within perimeter-4 rect."""
    # Check all radii positive
    if np.any(r <= 0):
        return False
    # Check pairwise non-overlap (disjoint interiors: dist >= r_i + r_j)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    # Check bounding rectangle perimeter <= 4
    w, h = _perimeter_constraint(x, y, r)
    if 2 * (w + h) > 4.0 + tol:
        return False
    return True

def _objective_and_grad(params, w, h):
    """
    params: [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    w, h: fixed rectangle dimensions (w + h = 2)
    Objective: -sum(r)  (we minimize negative sum)
    """
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]
    
    obj = -np.sum(r)
    
    # Gradient of objective
    grad = np.zeros(3*N)
    grad[2*N:] = -1.0
    
    # Constraints (inequality: g >= 0 means feasible)
    cons = []
    
    # Pairwise non-overlap: dist_ij >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12:
                d = 1e-12
                dx, dy = 1e-12, 0
            # g = d - r_i - r_j >= 0
            g = d - r[i] - r[j]
            # gradient of g
            gd = np.zeros(3*N)
            gd[i] = dx / d
            gd[j] = -dx / d
            gd[N + i] = dy / d
            gd[N + j] = -dy / d
            gd[2*N + i] = -1.0
            gd[2*N + j] = -1.0
            cons.append({'type': 'ineq', 'fun': lambda _, g=g: g, 'jac': lambda _: gd})
    
    # Boundary constraints: x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
    for i in range(N):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda _, i=i: params[i] - params[2*N+i], 
                     'jac': lambda _, i=i: _boundary_grad(i, 0)})
        # x_i + r_i <= w  =>  w - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda _, i=i: w - params[i] - params[2*N+i],
                     'jac': lambda _, i=i: _boundary_grad(i, 1)})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda _, i=i: params[N+i] - params[2*N+i],
                     'jac': lambda _, i=i: _boundary_grad(i, 2)})
        # y_i + r_i <= h  =>  h - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda _, i=i: h - params[N+i] - params[2*N+i],
                     'jac': lambda _, i=i: _boundary_grad(i, 3)})
    
    return obj, grad, cons

def _boundary_grad(i, which):
    g = np.zeros(3*N)
    if which == 0:  # x_i - r_i
        g[i] = 1.0
        g[2*N+i] = -1.0
    elif which == 1:  # w - x_i - r_i
        g[i] = -1.0
        g[2*N+i] = -1.0
    elif which == 2:  # y_i - r_i
        g[N+i] = 1.0
        g[2*N+i] = -1.0
    elif which == 3:  # h - y_i - r_i
        g[N+i] = -1.0
        g[2*N+i] = -1.0
    return g

def _solve_for_aspect(w, h, x0, y0, r0, max_iter=50, time_limit=20.0):
    """
    Solve packing in fixed rectangle w x h (w+h=2).
    Returns (x, y, r, success, sum_r).
    """
    params0 = np.concatenate([x0, y0, r0])
    
    best_params = params0.copy()
    best_sum_r = np.sum(r0)
    
    start_time = time.time()
    
    # Use SLSQP with multiple restarts
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb initial guess for restarts > 0
        if restart == 0:
            p = params0.copy()
        else:
            p = params0 + 0.1 * np.random.randn(3*N)
            p[2*N:] = np.maximum(p[2*N:], 0.01)
        
        # Clip positions to be within rectangle
        p[:N] = np.clip(p[:N], 0, w)
        p[N:2*N] = np.clip(p[N:2*N], 0, h)
        
        def fun(p):
            x = p[:N]
            y = p[N:2*N]
            r = p[2*N:]
            # Penalty for violations to help optimizer
            pen = 0.0
            # Pairwise overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    viol = r[i] + r[j] - d
                    if viol > 0:
                        pen += 100.0 * viol**2
            # Boundary penalty
            for i in range(N):
                if x[i] - r[i] < 0:
                    pen += 100.0 * (-(x[i]-r[i]))**2
                if x[i] + r[i] > w:
                    pen += 100.0 * (x[i]+r[i]-w)**2
                if y[i] - r[i] < 0:
                    pen += 100.0 * (-(y[i]-r[i]))**2
                if y[i] + r[i] > h:
                    pen += 100.0 * (y[i]+r[i]-h)**2
            return -np.sum(r) + pen
        
        def jac(p):
            x = p[:N]
            y = p[N:2*N]
            r = p[2*N:]
            g = np.zeros(3*N)
            g[2*N:] = -1.0
            
            # Pairwise gradient
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-10:
                        d = 1e-10
                        dx, dy = 1e-10, 0
                    viol = r[i] + r[j] - d
                    if viol > 0:
                        g[i] += 200.0 * viol * (dx/d)
                        g[j] += 200.0 * viol * (-dx/d)
                        g[N+i] += 200.0 * viol * (dy/d)
                        g[N+j] += 200.0 * viol * (-dy/d)
                        g[2*N+i] += 200.0 * viol * (-1)
                        g[2*N+j] += 200.0 * viol * (-1)
            
            # Boundary gradient
            for i in range(N):
                if x[i] - r[i] < 0:
                    g[i] += 200.0 * (-(x[i]-r[i])) * 1
                    g[2*N+i] += 200.0 * (-(x[i]-r[i])) * (-1)
                if x[i] + r[i] > w:
                    g[i] += 200.0 * (x[i]+r[i]-w) * 1
                    g[2*N+i] += 200.0 * (x[i]+r[i]-w) * 1
                if y[i] - r[i] < 0:
                    g[N+i] += 200.0 * (-(y[i]-r[i])) * 1
                    g[2*N+i] += 200.0 * (-(y[i]-r[i])) * (-1)
                if y[i] + r[i] > h:
                    g[N+i] += 200.0 * (y[i]+r[i]-h) * 1
                    g[2*N+i] += 200.0 * (y[i]+r[i]-h) * 1
            
            return g
        
        try:
            res = minimize(fun, p, jac=jac, method='SLSQP',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})
            if res.success or (hasattr(res, 'fun') and res.fun < -best_sum_r + 1e-6):
                p_opt = res.x
                s_r = np.sum(p_opt[2*N:])
                if s_r > best_sum_r:
                    best_sum_r = s_r
                    best_params = p_opt.copy()
        except Exception:
            pass
        
        if time.time() - start_time > time_limit:
            break
    
    x_opt = best_params[:N]
    y_opt = best_params[N:2*N]
    r_opt = best_params[2*N:]
    
    # Post-processing: fix any small violations
    x_opt, y_opt, r_opt = _fix_violations(x_opt, y_opt, r_opt, w, h)
    sum_r = np.sum(r_opt)
    
    return x_opt, y_opt, r_opt, True, sum_r

def _fix_violations(x, y, r, w, h, tol=1e-8):
    """Fix small violations by shrinking radii and adjusting positions."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Clip positions
    for i in range(N):
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])
    
    # Iteratively fix overlaps by shrinking radii
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    # Shrink both radii
                    excess = (r[i] + r[j] - d) / 2.0
                    r[i] = max(r[i] - excess, 0.001)
                    r[j] = max(r[j] - excess, 0.001)
                    changed = True
        # Fix boundary
        for i in range(N):
            if x[i] < r[i]:
                r[i] = max(x[i] - 1e-6, 0.001)
                changed = True
            if x[i] > w - r[i]:
                r[i] = max(w - x[i] - 1e-6, 0.001)
                changed = True
            if y[i] < r[i]:
                r[i] = max(y[i] - 1e-6, 0.001)
                changed = True
            if y[i] > h - r[i]:
                r[i] = max(h - y[i] - 1e-6, 0.001)
                changed = True
        if not changed:
            break
    
    return x, y, r

def _make_grid_init(w, h):
    """Create a grid-based initial placement."""
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % side
        row = i // side
        x[i] = (col + 0.5) * w / side
        y[i] = (row + 0.5) * h / side
        # Initial radius: half the cell size minus small gap
        cell_w = w / side
        cell_h = h / side
        r[i] = 0.45 * min(cell_w, cell_h)
    
    return x, y, r

def _make_random_init(w, h, seed=42):
    """Create a random initial placement with reasonable radii."""
    np.random.seed(seed)
    x = np.random.uniform(0.1*w, 0.9*w, N)
    y = np.random.uniform(0.1*h, 0.9*h, N)
    # Start with relatively large circles
    r = np.random.uniform(0.03, 0.08, N)
    
    # Try to separate them a bit
    for _ in range(50):
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-10:
                    push = (min_d - d) / 2.0 + 0.005
                    x[i] += push * dx / d
                    y[i] += push * dy / d
                    x[j] -= push * dx / d
                    y[j] -= push * dy / d
        # Keep in bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    return x, y, r

def construct():
    start_time = time.time()
    
    best_x = None
    best_y = None
    best_r = None
    best_sum = 0.0
    
    # Try several aspect ratios (w + h = 2, so h = 2 - w)
    # Perimeter = 2(w+h) = 4 always
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 1.3, 0.7, 1.1, 1.4, 0.9, 1.6]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Multiple initialization strategies
        inits = []
        
        # Grid init
        xg, yg, rg = _make_grid_init(w, h)
        inits.append((xg, yg, rg))
        
        # Random inits with different seeds
        for seed in [42, 123, 456, 789, 101]:
            if time.time() - start_time > DEADLINE - 5:
                break
            xr, yr, rr = _make_random_init(w, h, seed=seed)
            inits.append((xr, yr, rr))
        
        for x0, y0, r0 in inits:
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Allocate time based on remaining budget
            remaining = DEADLINE - 3 - (time.time() - start_time)
            time_limit = min(10.0, remaining / 2.0)
            if time_limit < 1.0:
                break
            
            x_opt, y_opt, r_opt, success, sum_r = _solve_for_aspect(
                w, h, x0, y0, r0, max_iter=30, time_limit=time_limit
            )
            
            if success and sum_r > best_sum:
                # Verify validity
                if _check_validity(x_opt, y_opt, r_opt):
                    best_sum = sum_r
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
                    best_r = r_opt.copy()
    
    # Final check
    if best_x is None:
        # Fallback: simple grid
        side = 5
        w = 1.0
        h = 1.0
        best_x = np.zeros(N)
        best_y = np.zeros(N)
        best_r = np.zeros(N)
        for i in range(N):
            col = i % side
            row = i // side
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 1e-3
    
    # Final validation and fix
    w_final = np.max(best_x + best_r) - np.min(best_x - best_r)
    h_final = np.max(best_y + best_r) - np.min(best_y - best_r)
    
    if 2 * (w_final + h_final) > 4.0 + 1e-6:
        # Scale down
        scale = 4.0 / (2.0 * (w_final + h_final)) * 0.999
        best_x = best_x * scale
        best_y = best_y * scale
        best_r = best_r * scale
    
    # Final validity check
    if not _check_validity(best_x, best_y, best_r):
        # Last resort: shrink all radii
        for i in range(N):
            best_r[i] *= 0.95
        if not _check_validity(best_x, best_y, best_r):
            # Fallback grid
            side = 5
            best_x = np.zeros(N)
            best_y = np.zeros(N)
            best_r = np.zeros(N)
            for i in range(N):
                col = i % side
                row = i // side
                best_x[i] = (col + 0.5) / side
                best_y[i] = (row + 0.5) / side
                best_r[i] = 0.5 / side - 1e-3
    
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
# EVOLVE-BLOCK-END
