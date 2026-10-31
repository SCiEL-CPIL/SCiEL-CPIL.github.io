import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Initial guess: Half of a 4x4 grid
    # Grid coords: -1.5, -0.5, 0.5, 1.5
    coords = [-1.5, -0.5, 0.5, 1.5]
    grid = np.array(list(itertools.product(coords, coords)))
    # Center grid
    # It is already centered at 0
    
    # Select 8 points for symmetry
    # Points with x > 0
    mask = grid[:, 0] > 0
    init_points = grid[mask]
    
    # Flatten
    init_params = init_points.flatten()
    
    best_score = np.inf
    best_params = init_params
    
    # Define soft objective
    def get_soft_objective(T):
        def obj(params):
            # Check time
            if time.time() > deadline - 5:
                return 1e10
            
            pts = params.reshape(8, 2)
            full_pts = np.vstack([pts, -pts])
            
            # Pairwise squared distances
            diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            
            # Mask diagonal
            mask = np.ones((16, 16), dtype=bool)
            np.fill_diagonal(mask, False)
            d_masked = dists_sq[mask]
            
            # If any distance is 0 (points coincide), return large value
            if np.min(d_masked) < 1e-8:
                return 1e10
            
            # Soft Max
            m_max = np.max(d_masked)
            # Prevent overflow in exp
            # exp(T * (d - m_max)) <= 1
            soft_max = m_max + np.log(np.sum(np.exp(T * (d_masked - m_max)))) / T
            
            # Soft Min
            m_min = np.min(d_masked)
            # exp(-T * (d - m_min)) <= 1
            soft_min = m_min - np.log(np.sum(np.exp(-T * (d_masked - m_min)))) / T
            
            if soft_min <= 0:
                return 1e10
                
            return soft_max / soft_min
        
        return obj

    # Annealing schedule
    temperatures = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_params = init_params
    
    for T in temperatures:
        if time.time() > deadline - 10:
            break
            
        obj_func = get_soft_objective(T)
        
        # Try to optimize
        try:
            res = minimize(obj_func, current_params, method='BFGS', 
                           options={'maxiter': 1000, 'gtol': 1e-8})
            if res.success or res.fun < best_score:
                current_params = res.x
                # Evaluate true objective for tracking
                # Note: res.fun is soft objective, not true ratio
                # We need to compute true ratio for best_score comparison
                pts = current_params.reshape(8, 2)
                full_pts = np.vstack([pts, -pts])
                diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=-1)
                mask = np.ones((16, 16), dtype=bool)
                np.fill_diagonal(mask, False)
                d_masked = dists_sq[mask]
                true_val = np.max(d_masked) / np.min(d_masked)
                
                if true_val < best_score:
                    best_score = true_val
                    best_params = current_params
        except Exception:
            pass
            
    # Final polish with high T or exact objective
    if time.time() < deadline - 5:
        def exact_obj(params):
            pts = params.reshape(8, 2)
            full_pts = np.vstack([pts, -pts])
            diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            mask = np.ones((16, 16), dtype=bool)
            np.fill_diagonal(mask, False)
            d_masked = dists_sq[mask]
            if np.min(d_masked) < 1e-8:
                return 1e10
            return np.max(d_masked) / np.min(d_masked)
        
        try:
            res = minimize(exact_obj, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_params = res.x
        except Exception:
            pass

    # Return best params as full 16 points
    pts = best_params.reshape(8, 2)
    full_pts = np.vstack([pts, -pts])
    return full_pts

# EVOLVE-BLOCK-END
