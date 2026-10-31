"""Improved circle packing: continuous optimization with multiple aspect ratios."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START

def _objective(params, W, H):
    """Objective: minimize sum of squared violations + negative sum of radii."""
    n = len(params)
    cx = params[0:n]
    cy = params[n:2*n]
    r = params[2*n:3*n]
    
    # Boundary violations
    bound_viol = 0.0
    for i in range(n):
        if cx[i] < r[i]:
            bound_viol += (r[i] - cx[i])**2
        if cx[i] > W - r[i]:
            bound_viol += (cx[i] - (W - r[i]))**2
        if cy[i] < r[i]:
            bound_viol += (r[i] - cy[i])**2
        if cy[i] > H - r[i]:
            bound_viol += (cy[i] - (H - r[i]))**2
    
    # Overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)**2
    
    # Maximize sum of radii (minimize negative)
    radius_term = -np.sum(r)
    
    return 1000.0 * (bound_viol + overlap_viol) + radius_term


def _feasibility_check(cx, cy, r, W, H, tol=1e-6):
    """Check if packing is feasible."""
    for i in range(len(r)):
        if cx[i] < r[i] - tol or cx[i] > W - r[i] + tol:
            return False
        if cy[i] < r[i] - tol or cy[i] > H - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _polish(cx, cy, r, W, H, max_iter=200):
    """Local polish using gradient-free method."""
    n = len(r)
    params = np.concatenate([cx, cy, r])
    
    def obj(p):
        return _objective(p, W, H)
    
    best_params = params.copy()
    best_val = obj(params)
    
    # Simple hill climbing with decreasing step size
    step = 0.01
    for it in range(max_iter):
        improved = False
        for dim in range(3*n):
            for sign in [1, -1]:
                trial = best_params.copy()
                trial[dim] += sign * step
                val = obj(trial)
                if val < best_val:
                    best_params = trial
                    best_val = val
                    improved = True
        if not improved:
            step *= 0.7
            if step < 1e-8:
                break
    
    return best_params[:n], best_params[n:2*n], best_params[2*n:3*n], best_val


def _solve_for_aspect(W, H, seed=0, time_limit=10.0):
    """Solve packing for given aspect ratio."""
    np.random.seed(seed)
    n = N
    max_r = min(W, H) / 2
    
    # Initial guess: grid placement
    cols = 5
    rows = 5
    cx_init = np.array([(i + 0.5) * W / cols for i in range(cols) for j in range(rows)][:n])
    cy_init = np.array([(j + 0.5) * H / rows for i in range(cols) for j in range(rows)][:n])
    r_init = np.full(n, min(W, H) / (2 * max(cols, rows)) * 0.9)
    
    params_init = np.concatenate([cx_init, cy_init, r_init])
    
    bounds = []
    for i in range(n):
        bounds.append((0, W))
    for i in range(n):
        bounds.append((0, H))
    for i in range(n):
        bounds.append((0, max_r))
    
    start_time = time.time()
    
    # Try multiple local optimizations from different starts
    best_params = params_init.copy()
    best_val = _objective(params_init, W, H)
    
    # Differential evolution for global search (limited)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        try:
            de_result = differential_evolution(
                lambda p: _objective(p, W, H),
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                seed=seed,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Local polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        cx_p, cy_p, r_p, val_p = _polish(
            best_params[:n], best_params[n:2*n], best_params[2*n:3*n],
            W, H, max_iter=int(remaining * 50)
        )
        if val_p < best_val:
            best_val = val_p
            best_params = np.concatenate([cx_p, cy_p, r_p])
    
    return best_params, best_val


def construct():
    """Construct 21 circles in a perimeter-4 rectangle maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.667, 2.0, 0.5, 1.333, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        # Verify perimeter
        perim = 2 * (W + H)
        if abs(perim - 4.0) > 1e-9:
            continue
        
        try:
            params, val = _solve_for_aspect(W, H, seed=int(ar*1000), time_limit=10.0)
            
            cx = params[:N]
            cy = params[N:2*N]
            r = params[2*N:3*N]
            
            # Check feasibility
            if _feasibility_check(cx, cy, r, W, H, tol=1e-5):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([cx, cy, r])
        except Exception:
            continue
    
    # Fallback: simple grid if nothing worked
    if best_result is None:
        W, H = 1.0, 1.0
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) * W / side
                y = (j + 0.5) * H / side
                rad = min(W, H) / (2 * side) - 1e-4
                cells.append([x, y, rad])
                count += 1
        best_result = np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
