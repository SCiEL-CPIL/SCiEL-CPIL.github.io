import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n_points = 14
    dim = 3
    n_vars = n_points * dim
    
    start_time = time.time()
    timeout = 600.0
    safety_margin = 10.0
    
    def objective(flat_points):
        points = flat_points.reshape(n_points, dim)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 0.0
        
        ratio = d_min / d_max
        # We want to maximize ratio^2, so minimize -ratio^2
        return -(ratio ** 2)

    # Initial bounds
    bounds = [(-1.0, 1.0)] * n_vars
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution
    # Use a moderate population and iteration count to stay within time limits
    # while exploring the space well.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.9,
            polish=False, # We will do manual polish
            init='latinhypercube'
        )
        
        if res.success or res.fun < best_score:
            best_score = -res.fun
            best_points = res.x.copy()
            
    except Exception as e:
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin:
        if best_points is None:
            return np.random.randn(n_points, dim)
        return best_points.reshape(n_points, dim)

    # Phase 2: Local Refinement with Nelder-Mead
    if best_points is not None:
        try:
            res_nm = minimize(
                objective, 
                best_points, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.copy()
                
        except Exception:
            pass

    # Final check
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin:
        pass # Just proceed to return

    if best_points is None:
        # Fallback to random if something went wrong
        return np.random.randn(n_points, dim)
    
    return best_points.reshape(n_points, dim)
