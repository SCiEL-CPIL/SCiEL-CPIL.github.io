# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import differential_evolution

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_soft_score(x, beta):
    """
    Objective for differential_evolution.
    Minimizes -R_soft.
    R_soft is a smooth approximation of (d_min / d_max)^2.
    """
    pts = x.reshape(16, 2)
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return 0.0
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # Soft max: 1/beta * log(sum(exp(beta * d)))
    
    # Numerical stability
    neg_d = -beta * dists
    max_neg_d = np.max(neg_d)
    log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
    soft_min = -log_sum_exp_neg / beta
    
    pos_d = beta * dists
    max_pos_d = np.max(pos_d)
    log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
    soft_max = log_sum_exp_pos / beta
    
    if soft_max < 1e-10:
        return 0.0
        
    ratio = soft_min / soft_max
    r_soft = ratio ** 2
    return -r_soft

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # We optimize a smooth surrogate.
    # beta controls the smoothness. Higher beta is closer to true score but harder to optimize.
    # We will run a few rounds with increasing beta, or just pick a good one.
    # For 16 points, beta ~ 10-50 is usually good for DE.
    beta = 20.0
    
    # Bounds: [-1, 1] for each coordinate.
    # Since the problem is scale-invariant, we can fix the scale.
    # However, DE works better with bounded boxes.
    # Normalizing the output will handle the scale.
    bounds = [(-1, 1)] * 32
    
    # Differential Evolution parameters
    # popsize=15, tol=1e-9, maxiter=1000
    # Since we have a time limit, we'll use a callback to stop if needed.
    # But DE in scipy is fairly efficient.
    # Let's run it.
    
    # We might want to try multiple betas or restarts if time permits.
    # But DE is global, so one good run is better than many local runs.
    
    try:
        # Run DE
        # seed=42 for reproducibility if needed, but DE is stochastic anyway.
        result = differential_evolution(
            _neg_soft_score, 
            bounds, 
            args=(beta,),
            maxiter=1000, 
            tol=1e-10,
            popsize=20,
            seed=42,
            polish=True,  # Use L-BFGS-B at the end for precision
            disp=False
        )
        
        best_points = result.x.reshape(16, 2)
        
    except Exception:
        # Fallback to a heuristic if DE fails
        # Simple grid or random
        best_points = np.random.randn(16, 2)
        
    # Normalize the best points found
    # The metric is scale invariant, but returning points in a reasonable range is good practice.
    # Also, the evaluator might check for finite values.
    best_points = best_points - best_points.mean(axis=0)
    norm = np.max(np.linalg.norm(best_points, axis=1))
    if norm > 1e-10:
        best_points = best_points / norm
        
    # If we have time left, we can try to polish with a higher beta or different method
    current_time = time.time()
    if current_time < deadline - 20:
        # Try a second run with higher beta to refine
        # Or just run DE again with a different seed
        # Let's try a higher beta for a more accurate local optimum near the global one
        try:
            result2 = differential_evolution(
                _neg_soft_score, 
                bounds, 
                args=(50.0,),
                maxiter=500, 
                tol=1e-10,
                popsize=15,
                seed=123,
                polish=True,
                disp=False
            )
            
            pts2 = result2.x.reshape(16, 2)
            # Compare scores
            s1 = _score(best_points)
            s2 = _score(pts2)
            
            if s2 > s1:
                best_points = pts2
                best_points = best_points - best_points.mean(axis=0)
                norm = np.max(np.linalg.norm(best_points, axis=1))
                if norm > 1e-10:
                    best_points = best_points / norm
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
