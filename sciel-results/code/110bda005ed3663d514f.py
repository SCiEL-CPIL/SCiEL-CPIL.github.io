"""Root finder for 3 fixed 2D systems using global derivative-free optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Objective function: minimize 0.5 * ||F(x)||^2
    def objective(x):
        # Check time limit
        if time.time() > deadline:
            return np.inf
        try:
            Fx = F(x)
            # Ensure finite result
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return 0.5 * np.dot(Fx, Fx)
        except Exception:
            return np.inf

    # Gradient estimation for BFGS (optional, but BFGS can do FD internally).
    # Since we want to save calls, we let BFGS do its own finite differences.
    # However, BFGS FD uses 1 call per dimension (2 calls) per iteration.
    # A pure derivative-free method like Powell or Nelder-Mead might be better for call count?
    # BFGS is usually faster in iterations. Let's stick to BFGS with analytic gradient if possible?
    # No, we don't have analytic gradient.
    # BFGS with finite differences: ~2-3 calls per iteration.
    # Powell: ~1 call per dimension per step? No, Powell is more expensive.
    # Nelder-Mead: ~1 call per vertex per step (3 vertices = 3 calls).
    # Actually, for 2D, BFGS is very efficient.
    
    x0_arr = np.array(x0, dtype=float)
    
    best_x = x0_arr.copy()
    best_val = objective(x0_arr)
    
    # Multi-start strategy: if initial point is bad, try a few random nearby points
    # But we must be careful with call count.
    # Let's first try the given x0.
    
    try:
        res = minimize(objective, x0_arr, method='BFGS', 
                       options={'gtol': 1e-10, 'maxiter': 100, 'disp': False})
        
        if res.success and np.all(np.isfinite(res.x)):
            x_candidate = res.x
            val_candidate = objective(x_candidate)
            if val_candidate < best_val:
                best_x = x_candidate
                best_val = val_candidate
        
        # Check if we have a good root
        # ||F(x)||^2 <= (1e-8)^2 = 1e-16
        # So objective <= 0.5e-16 = 5e-17
        if best_val <= 5e-17:
            return best_x
            
    except Exception:
        pass

    # If BFGS from x0 didn't work, try a few other starts
    # Random starts in a box around x0
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        # Random perturbation
        scale = 1.0
        x_try = x0_arr + rng.uniform(-scale, scale, size=2)
        
        try:
            res = minimize(objective, x_try, method='BFGS', 
                           options={'gtol': 1e-10, 'maxiter': 50, 'disp': False})
            
            if res.success and np.all(np.isfinite(res.x)):
                x_candidate = res.x
                val_candidate = objective(x_candidate)
                if val_candidate < best_val:
                    best_x = x_candidate
                    best_val = val_candidate
                    
                if best_val <= 5e-17:
                    return best_x
                    
        except Exception:
            continue

    return best_x
# EVOLVE-BLOCK-END
