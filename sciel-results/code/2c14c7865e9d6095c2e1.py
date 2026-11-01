import numpy as np
import scipy.spatial as spat
import time
from scipy.optimize import minimize

N = 14

def calc_score_fast(points):
    """
    Fast calculation of min triangle area / hull area.
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    n = len(points)
    # Generate all combinations of 3 indices
    # For N=14, C(14,3) = 364.
    
    # Vectorized triangle area calculation
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smoothed_neg_score(points, beta):
    """
    Computes the negative of the smoothed minimum triangle area normalized by hull area.
    Uses log-sum-exp to approximate the min function smoothly.
    We minimize -min_area, so we maximize min_area.
    To handle the non-differentiability of min, we use:
    min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
    Then we maximize this, which is equivalent to minimizing its negative.
    
    However, the hull area also depends on points.
    Objective: f(x) = min_triangle_area(x) / hull_area(x)
    We want to maximize f.
    
    Let A_min = min(areas)
    Let A_hull = hull_area
    
    Smoothed A_min: A_smooth = -1/beta * log(sum(exp(-beta * areas)))
    
    We minimize: -A_smooth / A_hull
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e10, np.zeros(points.size)
    except Exception:
        return 1e10, np.zeros(points.size)
    
    n = len(points)
    # Precomputed indices for speed
    # We can't easily precompute indices inside the function if it's called many times
    # unless we pass them or use a global. Let's use a global cache for indices.
    if not hasattr(_smoothed_neg_score, 'idx_cache'):
         i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
         mask = (i_idx < j_idx) & (j_idx < k_idx)
         _smoothed_neg_score.idx_cache = (i_idx[mask], j_idx[mask], k_idx[mask])
    
    i_idx, j_idx, k_idx = _smoothed_neg_score.idx_cache
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp trick for stability
    # We want -1/beta * log(sum(exp(-beta * areas)))
    # Let m = min(areas)
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # -1/beta * log(...) = m - 1/beta * log(sum(exp(-beta * (areas - m))))
    
    m = np.min(areas)
    shifted = areas - m
    # beta is a parameter. Larger beta -> closer to true min.
    # Smaller beta -> smoother.
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    a_smooth = m - (1.0 / beta) * log_sum_exp
    
    # Score to maximize is a_smooth / hull_area
    # We minimize negative score
    val = -a_smooth / hull_area
    return val, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points_init.copy()
    best_score = calc_score_fast(best_points)
    
    # Try multiple beta values for the smoothing, starting rough and getting finer
    # Or just use a moderate beta.
    # The gradient of the smoothed min is well-defined.
    
    # Since the problem is non-convex, a single run might get stuck.
    # We can try a few different initializations or beta values if time permits.
    
    # Strategy: 
    # 1. Run L-BFGS-B with a moderate beta (e.g., 10.0) to get close to a local optimum.
    # 2. If time remains, try a higher beta (e.g., 50.0) starting from the result of step 1.
    # 3. Also try a few random perturbations of the regular polygon to escape symmetry if needed.
    
    # Note: L-BFGS-B requires the function to return (f, g). 
    # We can use analytic gradient or numerical. 
    # Calculating analytic gradient of log-sum-exp min area / hull area is complex.
    # Numerical gradient is safer but slower. 
    # Given N=14, 28 variables. Numerical gradient takes 29 evaluations.
    # Each evaluation is fast (vectorized).
    # Let's try numerical gradient first for reliability.
    
    def objective(x):
        p = x.reshape(N, 2)
        val, _ = _smoothed_neg_score(p, beta=10.0)
        return val
    
    # Run minimization
    x0 = points_init.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < best_score:
            cand_points = res.x.reshape(N, 2)
            cand_score = calc_score_fast(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points
    except Exception:
        pass
    
    # If time allows, refine with higher beta
    if time.time() < deadline - 10:
        # Use the best points found so far as new start
        x_start = best_points.flatten()
        
        def objective_refine(x):
            p = x.reshape(N, 2)
            val, _ = _smoothed_neg_score(p, beta=50.0)
            return val
            
        try:
            res2 = minimize(objective_refine, x_start, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if res2.success or res2.fun < best_score:
                cand_points = res2.x.reshape(N, 2)
                cand_score = calc_score_fast(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_points
        except Exception:
            pass

    # Optional: Try a few random restarts if time is plentiful
    # But regular polygon is a very strong start for this symmetric problem.
    
    return best_points
