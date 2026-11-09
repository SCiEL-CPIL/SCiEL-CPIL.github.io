import numpy as np
import time
from scipy.optimize import minimize

def _smooth_max(x, eps=1e-8):
    """Smooth approximation of max(0, x)."""
    return (x + np.sqrt(x*x + eps)) / 2.0 - np.sqrt(eps)/2.0

def _objective(params, n=10, tol=1e-7, penalty_weight=1000.0):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Clip sides to positive for calculation stability
    sides = np.maximum(sides, 1e-6)
    
    # Objective: -sum(sides) + penalties
    obj = -np.sum(sides)
    
    # Boundary Penalty
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        v = -tol - (cx - s/2)
        obj += penalty_weight * _smooth_max(v)**2
        v = (cx + s/2) - (1 + tol)
        obj += penalty_weight * _smooth_max(v)**2
        v = -tol - (cy - s/2)
        obj += penalty_weight * _smooth_max(v)**2
        v = (cy + s/2) - (1 + tol)
        obj += penalty_weight * _smooth_max(v)**2
        
    # Overlap Penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t <= 0:
                continue
            ox = t - dx
            oy = t - dy
            sox = _smooth_max(ox)
            soy = _smooth_max(oy)
            obj += penalty_weight * sox * soy
            
    return obj

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # 1. Hardcoded "2 Large, 8 Small" Layout (Sum = 3.0)
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    idx = 2
    for r in range(2):
        for c in range(4):
            c1[idx] = [0.125 + c*0.25, 0.125 + r*0.25]
            s1[idx] = 0.25
            idx += 1
    update_best(c1, s1)
    
    # 2. Hardcoded "4 Large, 6 Small" Layout (Sum = 3.0)
    # 4 Large (0.375) in corners?
    # Let's try:
    # Top Left: 0.375 at (0.1875, 0.8125)
    # Top Right: 0.375 at (0.8125, 0.8125)
    # Bottom Left: 0.375 at (0.1875, 0.1875)
    # Bottom Right: 0.375 at (0.8125, 0.1875)
    # Center strip: 6 small (0.25)
    # This might not fit.
    # Let's use the optimizer to find a good "4 large, 6 small" config.
    # Initial guess: 4 large in a 2x2 grid, 6 small in remaining space.
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    # 4 Large in 2x2 grid of size 0.5x0.5? No, side 0.375.
    # Place them in corners.
    c2[0] = [0.1875, 0.8125]; s2[0] = 0.375
    c2[1] = [0.8125, 0.8125]; s2[1] = 0.375
    c2[2] = [0.1875, 0.1875]; s2[2] = 0.375
    c2[3] = [0.8125, 0.1875]; s2[3] = 0.375
    
    # 6 Small (0.25) in the center.
    # The central hole is roughly 0.25 wide (1 - 0.375 - 0.375 = 0.25).
    # Height is 1.0.
    # We can fit 4 squares of 0.25 in a column in the center?
    # 4 * 0.25 = 1.0.
    # So 4 squares in the vertical center strip.
    # And 2 squares in the horizontal center strip?
    # This is getting complicated. Let's just initialize randomly near this structure.
    for i in range(4, 10):
        c2[i] = [0.5, 0.5] + np.random.uniform(-0.2, 0.2, 2)
        s2[i] = 0.25
    update_best(c2, s2)
    
    # 3. Optimization from the best hardcoded layout
    # We will run L-BFGS-B from multiple starts.
    
    starts = []
    starts.append(np.concatenate([c1.flatten(), s1]))
    starts.append(np.concatenate([c2.flatten(), s2]))
    
    # Add a random start
    np.random.seed(42)
    c_rand = np.random.uniform(0.2, 0.8, (n, 2))
    s_rand = np.random.uniform(0.2, 0.3, n)
    starts.append(np.concatenate([c_rand.flatten(), s_rand]))
    
    for init_params in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_objective, init_params, args=(n, 1e-7, 1000.0), method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if time.time() < deadline:
                params_opt = res.x
                centers_opt = params_opt[:2*n].reshape(n, 2)
                sides_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(centers_opt, sides_opt)
        except Exception:
            pass
        
    # 4. Refinement: if we have a good result, do a tighter optimization
    if best_centers is not None and best_score > 3.0 and time.time() < deadline - 10:
        try:
            params_refine = np.concatenate([best_centers.flatten(), best_sides])
            res = minimize(_objective, params_refine, args=(n, 1e-7, 5000.0), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if time.time() < deadline:
                params_opt = res.x
                centers_opt = params_opt[:2*n].reshape(n, 2)
                sides_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(centers_opt, sides_opt)
        except Exception:
            pass
        
    # 5. Random Restarts with Nelder-Mead for robustness
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the best known layout
        if best_centers is not None:
            cx_init = best_centers[:, 0] + np.random.normal(0, 0.03, n)
            cy_init = best_centers[:, 1] + np.random.normal(0, 0.03, n)
            sides_init = best_sides + np.random.normal(0, 0.02, n)
        else:
            cx_init = np.random.uniform(0.2, 0.8, n)
            cy_init = np.random.uniform(0.2, 0.8, n)
            sides_init = np.random.uniform(0.2, 0.3, n)
        
        cx_init = np.clip(cx_init, 0.05, 0.95)
        cy_init = np.clip(cy_init, 0.05, 0.95)
        sides_init = np.clip(sides_init, 0.1, 0.5)
        
        centers_init = np.stack([cx_init, cy_init], axis=1)
        params_r = np.concatenate([centers_init.flatten(), sides_init])
        
        res = minimize(_objective, params_r, args=(n, 1e-7, 1000.0), method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if time.time() > deadline:
            break
            
        params_opt = res.x
        centers_opt = params_opt[:2*n].reshape(n, 2)
        sides_opt = np.maximum(params_opt[2*n:], 1e-6)
        update_best(centers_opt, sides_opt)
        
    # 6. Fallback
    if best_centers is None:
        best_centers = c1
        best_sides = s1
        
    return best_centers, best_sides
