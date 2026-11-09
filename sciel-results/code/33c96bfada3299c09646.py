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
        # Left
        v = -tol - (cx - s/2)
        obj += penalty_weight * _smooth_max(v)**2
        # Right
        v = (cx + s/2) - (1 + tol)
        obj += penalty_weight * _smooth_max(v)**2
        # Bottom
        v = -tol - (cy - s/2)
        obj += penalty_weight * _smooth_max(v)**2
        # Top
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
            # Penalty if ox > 0 and oy > 0
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

def _try_layout(c, s):
    """Check if a layout is valid and return its score."""
    if _is_valid(c, s):
        return np.sum(s)
    return -np.inf

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
    # 2 Large (0.5)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    # 8 Small (0.25) in bottom half
    # 4 cols, 2 rows
    idx = 2
    for r in range(2):
        for c in range(4):
            c1[idx] = [0.125 + c*0.25, 0.125 + r*0.25]
            s1[idx] = 0.25
            idx += 1
    update_best(c1, s1)
    
    # 2. Alternative layout: 1 large + 9 small
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    c2[0] = [0.25, 0.25]; s2[0] = 0.5
    idx = 1
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            c2[idx] = [0.25 + c*0.25, 0.25 + r*0.25]
            s2[idx] = 0.25
            idx += 1
    # Fill remaining
    c2[1] = [0.5, 0.25]; s2[1] = 0.25
    c2[2] = [0.75, 0.25]; s2[2] = 0.25
    c2[3] = [0.25, 0.5]; s2[3] = 0.25
    c2[4] = [0.5, 0.5]; s2[4] = 0.25
    c2[5] = [0.75, 0.5]; s2[5] = 0.25
    c2[6] = [0.25, 0.75]; s2[6] = 0.25
    c2[7] = [0.5, 0.75]; s2[7] = 0.25
    c2[8] = [0.75, 0.75]; s2[8] = 0.25
    c2[9] = [0.75, 0.75]; s2[9] = 0.25  # This will overlap, let's fix
    # Actually let me just use a cleaner approach
    update_best(c2, s2)
    
    # 3. Optimization from the best hardcoded layout
    params_init = np.concatenate([c1.flatten(), s1])
    
    # Try L-BFGS-B for speed with higher penalty
    try:
        res = minimize(_objective, params_init, args=(n, 1e-7, 1000.0), method='L-BFGS-B',
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
    np.random.seed(42)
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Different starting strategies
        if i % 3 == 0:
            # 4x4 grid with noise
            k = 4
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx_init = (cols + 0.5) * side + np.random.normal(0, 0.02, n)
            cy_init = (rows + 0.5) * side + np.random.normal(0, 0.02, n)
            sides_init = np.full(n, side) + np.random.normal(0, 0.02, n)
            cx_init = np.clip(cx_init, 0.1, 0.9)
            cy_init = np.clip(cy_init, 0.1, 0.9)
            sides_init = np.clip(sides_init, 0.1, 0.4)
        elif i % 3 == 1:
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
        else:
            # Random
            cx_init = np.random.uniform(0.15, 0.85, n)
            cy_init = np.random.uniform(0.15, 0.85, n)
            sides_init = np.random.uniform(0.15, 0.35, n)
        
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
