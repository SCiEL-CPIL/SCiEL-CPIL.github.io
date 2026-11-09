import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

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

def _smooth_penalty(x, beta=100.0):
    """Smooth approximation of max(0, x) using log-sum-exp"""
    return np.logaddexp(0, beta * x) / beta

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * _smooth_penalty(1.0 - s)
            continue
        # Left boundary violation
        v_left = -tol - (cx - s/2)
        if v_left > 0:
            pen += 1e6 * _smooth_penalty(v_left)
        # Right boundary violation
        v_right = (cx + s/2) - (1 + tol)
        if v_right > 0:
            pen += 1e6 * _smooth_penalty(v_right)
        # Bottom boundary violation
        v_bottom = -tol - (cy - s/2)
        if v_bottom > 0:
            pen += 1e6 * _smooth_penalty(v_bottom)
        # Top boundary violation
        v_top = (cy + s/2) - (1 + tol)
        if v_top > 0:
            pen += 1e6 * _smooth_penalty(v_top)
            
    # Overlap penalties (smooth)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            vx = t - abs(dx)
            vy = t - abs(dy)
            sx = _smooth_penalty(vx)
            sy = _smooth_penalty(vy)
            pen += 1e5 * (sx * sy)
            
    return -np.sum(sides) + pen

def _generate_initial_configs(n=10):
    """Generate multiple diverse initial configurations"""
    configs = []
    
    # Config 1: 4 large corners + 6 small (original)
    s_large = 0.375
    s_small = 0.25
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [s_large/2, s_large/2]; s1[0] = s_large
    c1[1] = [1 - s_large/2, s_large/2]; s1[1] = s_large
    c1[2] = [s_large/2, 1 - s_large/2]; s1[2] = s_large
    c1[3] = [1 - s_large/2, 1 - s_large/2]; s1[3] = s_large
    c1[4] = [0.5, 0.125]; s1[4] = s_small
    c1[5] = [0.5, 0.375]; s1[5] = s_small
    c1[6] = [0.5, 0.625]; s1[6] = s_small
    c1[7] = [0.5, 0.875]; s1[7] = s_small
    c1[8] = [0.1875, 0.5]; s1[8] = s_small
    c1[9] = [0.8125, 0.5]; s1[9] = s_small
    configs.append((c1, s1))
    
    # Config 2: 5 medium squares in a cross pattern
    s_med = 0.30
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    # Center square
    c2[0] = [0.5, 0.5]; s2[0] = s_med
    # 4 around it
    c2[1] = [0.5, 0.5 - s_med]; s2[1] = s_med
    c2[2] = [0.5, 0.5 + s_med]; s2[2] = s_med
    c2[3] = [0.5 - s_med, 0.5]; s2[3] = s_med
    c2[4] = [0.5 + s_med, 0.5]; s2[4] = s_med
    # 4 corners
    s_corner = 0.22
    c2[5] = [s_corner/2, s_corner/2]; s2[5] = s_corner
    c2[6] = [1 - s_corner/2, s_corner/2]; s2[6] = s_corner
    c2[7] = [s_corner/2, 1 - s_corner/2]; s2[7] = s_corner
    c2[8] = [1 - s_corner/2, 1 - s_corner/2]; s2[8] = s_corner
    # One more
    c2[9] = [0.5, 0.15]; s2[9] = 0.2
    configs.append((c2, s2))
    
    # Config 3: Grid-like 3x3 + 1
    s_grid = 0.31
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            c3[idx] = [s_grid/2 + i * s_grid, s_grid/2 + j * s_grid]
            s3[idx] = s_grid
    c3[9] = [0.5, 0.5]; s3[9] = 0.1
    configs.append((c3, s3))
    
    # Config 4: 2 rows of 5
    s_row = 0.19
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(5):
        c4[i] = [s_row/2 + i * s_row, s_row/2]
        s4[i] = s_row
        c4[i+5] = [s_row/2 + i * s_row, 1 - s_row/2]
        s4[i+5] = s_row
    configs.append((c4, s4))
    
    # Config 5: Random perturbation of config 1
    c5, s5 = c1.copy(), s1.copy()
    np.random.seed(42)
    for i in range(n):
        c5[i, 0] += np.random.uniform(-0.02, 0.02)
        c5[i, 1] += np.random.uniform(-0.02, 0.02)
        s5[i] += np.random.uniform(-0.01, 0.01)
    s5 = np.maximum(s5, 0.05)
    configs.append((c5, s5))
    
    return configs

def _polish(centers, sides, n=10, time_budget=30):
    """High-precision polish with tighter tolerances"""
    params_init = np.concatenate([centers.flatten(), sides])
    start = time.time()
    
    try:
        # First pass: Powell with tight tolerances
        res = minimize(_objective, params_init, args=(n,), method='Powell',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                'maxfev': 50000})
        params_opt = res.x
        
        if time.time() - start > time_budget:
            return centers, sides
        
        c_opt = params_opt[:2*n].reshape(n, 2)
        s_opt = np.maximum(params_opt[2*n:], 1e-6)
        
        # Second pass: Nelder-Mead for final polish
        res2 = minimize(_objective, params_opt, args=(n,), method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-11,
                                 'maxfev': 20000})
        params_opt2 = res2.x
        c_opt2 = params_opt2[:2*n].reshape(n, 2)
        s_opt2 = np.maximum(params_opt2[2*n:], 1e-6)
        
        # Return the better result
        score1 = np.sum(s_opt) if _is_valid(c_opt, s_opt) else -np.inf
        score2 = np.sum(s_opt2) if _is_valid(c_opt2, s_opt2) else -np.inf
        
        if score2 > score1:
            return c_opt2, s_opt2
        else:
            return c_opt, s_opt
    except Exception:
        return centers, sides

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
    
    # Generate initial configurations
    configs = _generate_initial_configs(n)
    
    # Evaluate all initial configs
    for centers, sides in configs:
        update_best(centers, sides)
    
    # Optimize each configuration
    for idx, (centers, sides) in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        params_init = np.concatenate([centers.flatten(), sides])
        
        try:
            # Powell optimization
            res = minimize(_objective, params_init, args=(n,), method='Powell',
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'maxfev': 30000})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Final polish on the best solution
    if best_centers is not None and time.time() < deadline - 30:
        best_centers, best_sides = _polish(best_centers, best_sides, n, 
                                           time_budget=min(30, deadline - time.time() - 10))
    
    if best_centers is None:
        # Fallback to first config
        best_centers = configs[0][0]
        best_sides = configs[0][1]
    
    return best_centers, best_sides
