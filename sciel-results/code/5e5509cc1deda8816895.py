import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def log_score_from_points(P):
    """
    Calculate the log-score for a set of points P (8x2).
    Score = Delta / n^n, where Delta is product of ordered pair distances,
    after scaling P to diameter 2.
    Returns log(Score).
    """
    n = P.shape[0]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return -1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return -1e100
        
    # Scale to diameter 2
    scale = 2.0 / D
    # Delta is product of all ordered pairs (i != j)
    # log(Delta) = sum(log(dists))
    # We want Delta_scaled / n^n
    # Delta_scaled = Delta * scale^(n*(n-1))
    # log(Score) = log(Delta) + n*(n-1)*log(scale) - n*log(n)
    
    sum_log_dists = np.sum(np.log(dists))
    log_score = sum_log_dists + n*(n-1)*np.log(scale) - n*np.log(n)
    
    return log_score

def objective_de(x):
    """
    Objective for Differential Evolution.
    x: 16-d vector representing 8 points in 2D.
    Returns negative log-score.
    """
    P = x.reshape(8, 2)
    return -log_score_from_points(P)

def objective_lbfgs(x):
    """
    Objective for L-BFGS-B.
    x: 16-d vector representing 8 points in 2D.
    Returns negative log-score.
    """
    P = x.reshape(8, 2)
    return -log_score_from_points(P)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Baseline: Regular octagon
    base_angles = np.arange(8) * (2 * np.pi / 8)
    P_reg = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    best_val = log_score_from_points(P_reg)
    best_points = P_reg.copy()
    
    # Strategy: Global search with Differential Evolution, then local polish with L-BFGS-B
    
    # Define bounds for coordinates. 
    # Since we scale to diameter 2, the absolute scale doesn't matter for the score, 
    # but we need to keep coordinates within reasonable bounds to avoid numerical issues.
    # The regular octagon has radius 1. The optimal configuration is likely similar in scale.
    # Let's use bounds [-2, 2] for each coordinate.
    bounds = [(-2.0, 2.0)] * 16
    
    # Initial population for DE
    # We can seed the population with the regular octagon and some random perturbations
    # to help DE converge faster.
    
    try:
        # Run Differential Evolution
        # maxiter is set to allow sufficient exploration. 
        # tol is tight to ensure convergence.
        res_de = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-10, 
            popsize=15, 
            init='sobol'
        )
        
        if res_de.fun < 1e99:
            val_de = -res_de.fun
            if val_de > best_val:
                best_val = val_de
                best_points = res_de.x.reshape(8, 2).copy()
                
    except Exception:
        pass

    # Polish the best result found so far with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            # Use the best points from DE (or regular octagon if DE failed) as starting point
            x0 = best_points.flatten()
            
            res_lbfgs = minimize(
                objective_lbfgs, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res_lbfgs.fun < 1e99:
                val_lbfgs = -res_lbfgs.fun
                if val_lbfgs > best_val:
                    best_val = val_lbfgs
                    best_points = res_lbfgs.x.reshape(8, 2).copy()
                    
        except Exception:
            pass

    # Final check for validity and distinctness
    if best_points is None:
        return P_reg.tolist()
        
    # Ensure distinct points
    dists_sq = np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2)
    if np.any(dists_sq[~np.eye(8, dtype=bool)] < 1e-9):
        return P_reg.tolist()
        
    # Final score calculation to ensure we return the best valid set
    final_val = log_score_from_points(best_points)
    if final_val < log_score_from_points(P_reg):
         return P_reg.tolist()
         
    return best_points.tolist()
