import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_herm

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    def time_left():
        return deadline - time.time()
    
    H0_0 = 1.0
    H4_0 = float(eval_herm(4, 0))
    H8_0 = float(eval_herm(8, 0))
    H12_0 = float(eval_herm(12, 0))
    
    def get_a12(a0, a4, a8):
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Precompute grids
    x_coarse = np.linspace(-8, 8, 8001)
    x_fine = np.linspace(-10, 10, 40001)
    
    def objective(params, grid=None):
        if grid is None:
            grid = x_coarse
        a0, a4, a8 = params
        a12 = get_a12(a0, a4, a8)
        
        poly = (a0 * eval_herm(0, grid) + a4 * eval_herm(4, grid) + 
                a8 * eval_herm(8, grid) + a12 * eval_herm(12, grid))
        g = poly * np.exp(-grid**2 / 2)
        return np.trapz(np.abs(g), grid)
    
    def objective_fine(params):
        return objective(params, x_fine)
    
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = objective_fine(x0)
    best_params = x0.copy()
    
    # Phase 1: Coarse optimization
    try:
        result = minimize(lambda p: objective(p, x_coarse), x0, method='L-BFGS-B', 
                         options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < objective(x0, x_coarse):
            # Verify on fine grid
            if objective_fine(result.x) < best_val:
                best_val = objective_fine(result.x)
                best_params = result.x
    except Exception:
        pass
    
    if time_left() < 15.0:
        return np.array(best_params)
    
    # Phase 2: Fine optimization from best
    try:
        result = minimize(objective_fine, best_params, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time_left() < 20.0:
        return np.array(best_params)
    
    # Phase 3: Multi-start on coarse, then refine
    rng = np.random.RandomState(42)
    for i in range(40):
        if time_left() < 25.0:
            break
        scale = 2.0 / (1 + i * 0.15)
        x_try = best_params + rng.randn(3) * scale
        try:
            result = minimize(lambda p: objective(p, x_coarse), x_try, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < objective(best_params, x_coarse):
                val_fine = objective_fine(result.x)
                if val_fine < best_val:
                    best_val = val_fine
                    best_params = result.x
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time_left() > 30.0:
        try:
            result = minimize(objective_fine, best_params, method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return np.array(best_params)
