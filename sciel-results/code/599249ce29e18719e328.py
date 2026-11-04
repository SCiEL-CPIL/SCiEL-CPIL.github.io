"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]
_NUM_TRI = len(_TRI_I)

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Check for duplicates or invalid points
    if np.any(~np.isfinite(pts)):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _smooth_obj(params, tau):
    """
    Objective function for optimization.
    Minimizes the negative smooth minimum area ratio.
    params: (2N,) array of [x_1, y_1, ..., x_N, y_N]
    tau: sharpness parameter for logsumexp approximation of min
    """
    pts = params.reshape(N, 2)
    
    # Basic validity check
    if np.any(~np.isfinite(pts)):
        return 1e9

    # Compute Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e9
    except Exception:
        return 1e9

    # Compute all triangle areas using pre-computed indices
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    # We want to maximize min(ratios).
    # Minimize -min(ratios).
    # Approx -min(y) with (1/tau) * log(sum(exp(-tau * y)))
    
    # Numerical stability: subtract max of exponents (which is -tau * min(ratios))
    # max_exp = -tau * min(ratios)
    # sum_exp = sum(exp(-tau * ratios - max_exp)) = sum(exp(tau * (min(ratios) - ratios)))
    # log_sum = max_exp + log(sum_exp)
    # result = log_sum / tau = min(ratios) + (1/tau) * log(sum(exp(tau * (min(ratios) - ratios))))
    # Since ratios >= min(ratios), the terms inside exp are <= 0, so no overflow.
    
    min_ratio = np.min(ratios)
    # Shift to ensure stability
    exponents = -tau * (ratios - min_ratio)
    # exponents are <= 0
    
    sum_exp = np.sum(np.exp(exponents))
    
    if sum_exp == 0:
        return 1e9
        
    log_sum = np.log(sum_exp)
    
    # The value is min_ratio + (1/tau)*log_sum
    # We want to minimize the negative of this
    val = min_ratio + (log_sum / tau)
    
    return -val

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s
    time_limit = 140.0 
    
    # 1. Global Search using Differential Evolution
    # Bounds: We can constrain points to a box, e.g., [-2, 2] x [-2, 2]
    # The normalization by hull area makes the scale irrelevant, 
    # but DE works better with bounded search spaces.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # We use a smooth objective with a moderate tau for DE.
    # DE doesn't use gradients, but a smooth landscape helps the population move.
    tau_de = 20.0
    
    def de_objective(params):
        return _smooth_obj(params, tau_de)
    
    # Run DE
    # maxiter and popsize need to be balanced against time.
    # N=26 variables. 
    # popsize=15 (default is 15*D? No, default is 15. Let's use 20 for robustness).
    # maxiter=500 might be too slow. Let's monitor time.
    
    best_pts = None
    best_score = 0.0
    
    try:
        # Use a timeout callback? DE doesn't have a direct time callback in older scipy, 
        # but we can just run it and check time. 
        # Better: run DE for a fixed number of iterations, then polish.
        
        # To be safe, we run DE for a limited time or iterations.
        # Let's try 200 iterations with popsize 20.
        # Each iteration involves evaluating 20*26 points? No, 20 points per generation.
        # Each evaluation is O(N^3) ~ 286 ops. Very fast.
        
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5, 
            seed=42,
            init='sobol' # Better initial distribution
        )
        
        de_pts = res_de.x.reshape(N, 2)
        de_score = _compute_score(de_pts)
        
        if de_score > best_score:
            best_score = de_score
            best_pts = de_pts.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
        pass

    # 2. Local Polish with L-BFGS-B
    # Use multiple starting points for polishing:
    # a) The result from DE
    # b) A regular polygon
    # c) A few random points
    
    polish_starts = []
    
    if best_pts is not None:
        polish_starts.append(best_pts)
        
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    polish_starts.append(p_reg)
    
    # Add a few random starts
    rng = np.random.default_rng(123)
    for _ in range(3):
        p_rand = rng.uniform(-1, 1, (N, 2))
        polish_starts.append(p_rand)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
         return best_pts if best_pts is not None else p_reg

    # Polish each start
    # Use a high tau for sharpness
    tau_polish = 200.0
    
    for p0 in polish_starts:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
            
        def polish_obj(params):
            return _smooth_obj(params, tau_polish)
            
        try:
            res_polish = minimize(
                polish_obj, 
                p0.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            
            new_pts = res_polish.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            continue
            
    # 3. Final High-Precision Polish
    # If we have time, do one more polish with even higher tau on the best result
    elapsed = time.time() - start_time
    if best_pts is not None and elapsed < time_limit - 2:
        tau_final = 1000.0
        
        def final_obj(params):
            return _smooth_obj(params, tau_final)
            
        try:
            res_final = minimize(
                final_obj, 
                best_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'gtol': 1e-10}
            )
            
            new_pts = res_final.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        return p_reg
        
    return best_pts

# EVOLVE-BLOCK-END
