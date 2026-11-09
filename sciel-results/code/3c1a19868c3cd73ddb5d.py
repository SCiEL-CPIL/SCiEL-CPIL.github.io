import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
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
    
    # --- Initial Configurations ---
    
    # Config 1: 2 large (0.5), 8 small (0.25)
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
    update_best(c1, s1)
    
    # Config 2: 3x3 grid of 0.3 + center 0.1
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    idx = 0
    for r in range(3):
        for c in range(3):
            c2[idx] = [0.05 + (c + 0.5)*0.3, 0.05 + (r + 0.5)*0.3]
            s2[idx] = 0.3
            idx += 1
    c2[9] = [0.5, 0.5]
    s2[9] = 0.1
    update_best(c2, s2)
    
    # Random starts
    np.random.seed(42)
    starts = [(c1.copy(), s1.copy()), (c2.copy(), s2.copy())]
    
    for _ in range(15):
        c_r = c1.copy() + np.random.normal(0, 0.03, (n, 2))
        s_r = s1.copy() + np.random.normal(0, 0.02, n)
        s_r = np.clip(s_r, 0.05, 0.6)
        starts.append((c_r, s_r))
    
    for _ in range(15):
        c_r = c2.copy() + np.random.normal(0, 0.03, (n, 2))
        s_r = s2.copy() + np.random.normal(0, 0.02, n)
        s_r = np.clip(s_r, 0.05, 0.5)
        starts.append((c_r, s_r))
    
    for _ in range(30):
        c_r = np.random.uniform(0.1, 0.9, (n, 2))
        s_r = np.random.uniform(0.05, 0.45, n)
        starts.append((c_r, s_r))
    
    # --- Optimization with smooth soft-constraint loss ---
    
    def _soft_max_0(x, eps=1e-8):
        """Smooth max(0, x) with C1 continuity."""
        return (x + np.sqrt(x*x + eps*eps)) / 2.0 - eps / 2.0
    
    def _objective(params, n=10):
        """Objective: minimize -sum(sides) + soft penalty for overlaps and boundaries."""
        centers = params[:2*n].reshape(n, 2)
        sides = np.maximum(params[2*n:], 1e-4)
        
        # Primary term: maximize sum of sides
        obj = -np.sum(sides)
        
        # Overlap penalty (smooth)
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                if t > 0:
                    ox = t - dx
                    oy = t - dy
                    A = _soft_max_0(ox)
                    B = _soft_max_0(oy)
                    overlap_pen += A * B
        
        # Boundary penalty (smooth)
        boundary_pen = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            v_l = -1e-7 - (cx - s/2)
            v_r = (cx + s/2) - (1 + 1e-7)
            v_b = -1e-7 - (cy - s/2)
            v_t = (cy + s/2) - (1 + 1e-7)
            boundary_pen += _soft_max_0(v_l)**2
            boundary_pen += _soft_max_0(v_r)**2
            boundary_pen += _soft_max_0(v_b)**2
            boundary_pen += _soft_max_0(v_t)**2
        
        # Weight the penalties
        obj += 100.0 * overlap_pen
        obj += 100.0 * boundary_pen
        
        return obj
    
    def _is_valid(centers, sides, tol=1e-7):
        """Check if configuration is valid."""
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
    
    # Optimize each start
    for c_init, s_init in starts:
        if time.time() > deadline - 10:
            break
        
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            # Use L-BFGS-B for gradient-based optimization
            res = minimize(_objective, params_init, args=(n,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < _objective(params_init, n):
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-4)
                
                update_best(c_opt, s_opt)
                
                # Polish with SLSQP
                if _is_valid(c_opt, s_opt) and time.time() < deadline - 5:
                    res_polish = minimize(_objective, params_opt, args=(n,), method='SLSQP',
                                          options={'maxiter': 200, 'ftol': 1e-14, 'disp': False})
                    if res_polish.success:
                        params_pol = res_polish.x
                        c_pol = params_pol[:2*n].reshape(n, 2)
                        s_pol = np.maximum(params_pol[2*n:], 1e-4)
                        update_best(c_pol, s_pol)
                        
        except Exception:
            pass
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 20:
        params_best = np.concatenate([best_centers.flatten(), best_sides])
        
        try:
            res_final = minimize(_objective, params_best, args=(n,), method='SLSQP',
                                 options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            
            if res_final.success or res_final.fun < _objective(params_best, n):
                params_final = res_final.x
                c_final = params_final[:2*n].reshape(n, 2)
                s_final = np.maximum(params_final[2*n:], 1e-4)
                
                if _is_valid(c_final, s_final):
                    score_final = np.sum(s_final)
                    if score_final > best_score:
                        best_score = score_final
                        best_centers = c_final
                        best_sides = s_final
                        
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = c1
        best_sides = s1
    
    return best_centers, best_sides
