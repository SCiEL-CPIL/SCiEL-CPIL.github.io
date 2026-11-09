# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smooth_overlap_penalty(centers, sides, tol=1e-7, beta=50.0):
    n = len(sides)
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t <= 0:
                continue
            ox = t - dx
            oy = t - dy
            eps = 1e-12
            soft_A = (ox + np.sqrt(ox*ox + eps)) / 2.0 - np.sqrt(eps)/2.0
            soft_B = (oy + np.sqrt(oy*oy + eps)) / 2.0 - np.sqrt(eps)/2.0
            if soft_A > 0 and soft_B > 0:
                pen += soft_A * soft_B
    return pen

def _boundary_penalty(centers, sides, tol=1e-7):
    pen = 0.0
    for i in range(len(sides)):
        cx, cy = centers[i]
        s = sides[i]
        v_l = -tol - (cx - s/2)
        v_r = (cx + s/2) - (1 + tol)
        v_b = -tol - (cy - s/2)
        v_t = (cy + s/2) - (1 + tol)
        
        def smooth_max_0(x):
            eps = 1e-12
            return (x + np.sqrt(x*x + eps)) / 2.0 - np.sqrt(eps)/2.0
        
        pen += smooth_max_0(v_l)**2
        pen += smooth_max_0(v_r)**2
        pen += smooth_max_0(v_b)**2
        pen += smooth_max_0(v_t)**2
    return pen

def _objective(params, n=10):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    sides_clipped = np.maximum(sides, 1e-6)
    
    obj = -np.sum(sides_clipped)
    obj += 50.0 * _smooth_overlap_penalty(centers, sides_clipped)
    obj += 50.0 * _boundary_penalty(centers, sides_clipped)
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
    deadline = start_time + 580
    
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

    starts = []
    
    # Config 1: 2 large (0.5), 8 small (0.25) -> Sum 3.0
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    c1[2] = [0.125, 0.375]; s1[2] = 0.25
    c1[3] = [0.375, 0.375]; s1[3] = 0.25
    c1[4] = [0.125, 0.125]; s1[4] = 0.25
    c1[5] = [0.375, 0.125]; s1[5] = 0.25
    c1[6] = [0.625, 0.375]; s1[6] = 0.25
    c1[7] = [0.875, 0.375]; s1[7] = 0.25
    c1[8] = [0.625, 0.125]; s1[8] = 0.25
    c1[9] = [0.875, 0.125]; s1[9] = 0.25
    starts.append((c1, s1))
    
    # Config 2: 10 Equal Squares (Target ~3.07)
    # Start with a 3x3 grid of 0.3 and one in the middle?
    # 3x3 grid of 0.3: width 0.9.
    # Place them in 3x3 grid, and the 10th in the center gap?
    # Gap is 0.1. 0.3 doesn't fit.
    # So start with 10 random valid-ish positions of side 0.3.
    
    np.random.seed(42)
    
    # Generate 5 random starts with equal sides 0.3
    for k in range(5):
        c_eq = np.random.uniform(0.3, 0.7, (n, 2))
        s_eq = np.full(n, 0.3)
        starts.append((c_eq, s_eq))
        
    # Add perturbations of Config 1
    for _ in range(5):
        c_r = c1.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s1.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.1, 0.6)
        starts.append((c_r, s_r))
        
    # Pure random starts
    for _ in range(10):
        c_r = np.random.uniform(0.2, 0.8, (n, 2))
        s_r = np.random.uniform(0.2, 0.4, n)
        starts.append((c_r, s_r))

    for c_init, s_init in starts:
        if time.time() > deadline:
            break
            
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            res = minimize(_objective, params_init, args=(n,), method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-9, 'disp': False})
            
            if res.success or res.fun < _objective(params_init, n):
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                
                update_best(c_opt, s_opt)
                
                # Polish
                if _is_valid(c_opt, s_opt):
                    res_polish = minimize(_objective, params_opt, args=(n,), method='SLSQP',
                                          options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
                    if res_polish.success:
                        params_pol = res_polish.x
                        c_pol = params_pol[:2*n].reshape(n, 2)
                        s_pol = np.maximum(params_pol[2*n:], 1e-6)
                        update_best(c_pol, s_pol)
                        
        except Exception:
            pass

    if best_centers is None:
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides

# EVOLVE-BLOCK-END
