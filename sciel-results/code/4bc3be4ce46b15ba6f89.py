import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 60.0

def _raw_score(points):
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max < 1e-10:
        return 0.0
    if d_min < 1e-10:
        return 0.0
        
    ratio = d_min / d_max
    return ratio * ratio

def _objective_de(x_flat):
    points = x_flat.reshape(14, 3)
    score = _raw_score(points)
    return -score

def _soft_min_max_score(points, beta=100.0):
    """
    Calculate score using soft min/max to make the objective smooth.
    """
    dists = pdist(points)
    if np.any(dists < 1e-10):
        return 0.0
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # Soft max: 1/beta * log(sum(exp(beta * d)))
    # To avoid overflow, use log-sum-exp trick
    log_sum_exp_neg = np.log(np.sum(np.exp(-beta * dists)))
    log_sum_exp_pos = np.log(np.sum(np.exp(beta * dists)))
    
    d_min_soft = -1.0 / beta * log_sum_exp_neg
    d_max_soft = 1.0 / beta * log_sum_exp_pos
    
    if d_max_soft < 1e-10:
        return 0.0
        
    ratio = d_min_soft / d_max_soft
    return ratio * ratio

def _objective_soft(x_flat, beta=100.0):
    points = x_flat.reshape(14, 3)
    score = _soft_min_max_score(points, beta)
    return -score

def _polish(points0, deadline):
    """
    Refine points using Soft L-BFGS-B then Nelder-Mead.
    """
    x0 = points0.flatten()
    
    # 1. Soft L-BFGS-B
    if time.time() < deadline - 10:
        try:
            res_soft = minimize(
                lambda x: _objective_soft(x, beta=100.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            x0 = res_soft.x
        except Exception:
            pass

    # 2. Nelder-Mead on raw score
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                lambda x: _raw_score(x.reshape(14, 3)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            x0 = res_nm.x
        except Exception:
            pass
            
    return x0.reshape(14, 3)

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    
    best_score = 0.0
    best_points = np.random.default_rng(0).standard_normal((14, 3))
    
    bounds = [(-1.0, 1.0)] * 42
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
            
        try:
            res = differential_evolution(
                _objective_de,
                bounds,
                seed=seed,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            cand_points = res.x.reshape(14, 3)
            
            # Polish
            polished_points = _polish(cand_points, deadline)
            
            score = _raw_score(polished_points)
            if score > best_score:
                best_score = score
                best_points = polished_points
                
        except Exception:
            pass
            
    if time.time() < deadline - 5:
        polished_best = _polish(best_points, deadline)
        score = _raw_score(polished_best)
        if score > best_score:
            best_score = score
            best_points = polished_best

    if best_points.shape != (14, 3) or not np.all(np.isfinite(best_points)):
        best_points = np.random.default_rng(123).standard_normal((14, 3))
        
    return best_points
