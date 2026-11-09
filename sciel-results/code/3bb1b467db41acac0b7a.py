"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    def time_left():
        return time_limit - (time.time() - start_time)

    def objective(x):
        # Return squared norm to avoid sqrt cost, and handle non-finite values
        try:
            val = F(x)
            if not np.all(np.isfinite(val)):
                return 1e10
            return float(np.dot(val, val))
        except Exception:
            return 1e10

    x0 = np.array(x0, dtype=float)
    
    # Initial check
    try:
        f0 = F(x0)
        if np.all(np.isfinite(f0)) and np.linalg.norm(f0) <= 1e-8:
            return x0
    except Exception:
        pass

    best_x = x0
    best_val = objective(x0)
    
    # Generate deterministic multi-start points
    # 1. Original x0
    # 2. Perturbations in 4 cardinal directions with varying magnitudes
    # 3. A few random-looking deterministic points based on x0
    starts = [x0]
    for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for dx in [-scale, 0.0, scale]:
            for dy in [-scale, 0.0, scale]:
                if dx == 0.0 and dy == 0.0:
                    continue
                starts.append(x0 + np.array([dx, dy]))
    
    # Add a few "far" points to ensure global coverage if x0 is far from root
    starts.append(x0 + np.array([100.0, 100.0]))
    starts.append(x0 + np.array([-100.0, 100.0]))
    starts.append(x0 + np.array([100.0, -100.0]))
    starts.append(x0 + np.array([-100.0, -100.0]))
    
    # Shuffle starts deterministically to avoid biasing towards local minima near x0
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(starts))
    starts = [starts[i] for i in indices]

    # Nelder-Mead options
    # maxiter is high because each iteration is cheap (1 function call)
    # But we will break out based on time
    options = {
        'maxiter': 1000,
        'xatol': 1e-10,
        'fatol': 1e-16,
        'adaptive': True
    }

    for x_start in starts:
        if time_left() < 1.0:
            break
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', options=options)
            x_opt = res.x
            val_opt = res.fun
            
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
                
                # Check if we met the tolerance
                if val_opt <= (1e-8)**2:
                    return best_x
                    
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
