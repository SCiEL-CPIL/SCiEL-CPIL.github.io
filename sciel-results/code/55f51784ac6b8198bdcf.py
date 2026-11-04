"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    
    # Vectorized computation of all triangle areas
    # pts shape (n, 2)
    # We want areas for all i < j < k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # Get indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Extract points
    p_i = pts[i_idx]
    p_j = pts[j_idx]
    p_k = pts[k_idx]
    
    # Calculate cross products
    v1 = p_j - p_i
    v2 = p_k - p_i
    crosses = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    areas = 0.5 * crosses
    
    min_area = np.min(areas)
    return min_area / hull_area

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    Outer angles: 0, 2pi/7, ...
    Inner angles: theta, theta + 2pi/7, ...
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _soft_min_score(pts, beta=50.0):
    """
    Compute a smooth approximation of the score using soft-min for triangle areas.
    score = soft_min(areas) / hull_area
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    n = len(pts)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = pts[i_idx]
    p_j = pts[j_idx]
    p_k = pts[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    crosses = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    areas = 0.5 * crosses
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, subtract min
    min_a = np.min(areas)
    # If areas are very close, soft min is close to min
    # Use a stable log-sum-exp variant
    # soft_min = min_a - 1/beta * log(sum(exp(-beta * (areas - min_a))))
    # Note: exp(-beta * (areas - min_a)) <= 1, so sum is bounded by N_triples
    shifted = areas - min_a
    exp_term = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_term))
    soft_min_val = min_a - (1.0 / beta) * log_sum
    
    return soft_min_val / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    best_pts = _generate_two_ring_points(best_r, best_theta)
    
    # Strategy:
    # 1. Coarse Grid Search to find promising (r, theta)
    # 2. L-BFGS-B optimization on soft-min surrogate
    # 3. Final polish with exact score and small perturbations
    
    def _neg_soft_score(params):
        r, th = params
        if r < 0.1 or r > 0.99:
            return 1e9
        pts = _generate_two_ring_points(r, th)
        s = _soft_min_score(pts, beta=100.0)
        return -s

    # 1. Coarse Grid
    # theta range [0, pi/7] due to symmetry
    # r range [0.3, 0.95]
    
    theta_coarse = np.linspace(0, np.pi/7, 10)
    
    for th in theta_coarse:
        if time.time() > deadline - 30.0:
            break
            
        # Optimize r for this theta using soft-min
        def neg_score_r(r):
            if r < 0.1 or r > 0.99: return 1e9
            return _neg_soft_score([r, th])
            
        try:
            res_r = minimize(neg_score_r, x0=0.6, method='L-BFGS-B', bounds=[(0.2, 0.95)], options={'maxiter': 50, 'ftol': 1e-8})
            r_opt = res_r.x[0]
            score_opt = -res_r.fun
            
            # Verify with exact score
            pts_test = _generate_two_ring_points(r_opt, th)
            exact_s = _score(pts_test)
            
            if exact_s > best_score:
                best_score = exact_s
                best_r = r_opt
                best_theta = th
                best_pts = pts_test
        except Exception:
            pass

    # 2. L-BFGS-B on 2D (r, theta) using soft-min
    if time.time() < deadline - 20.0:
        # Start from best found so far
        x0 = [best_r, best_theta]
        
        try:
            res = minimize(_neg_soft_score, x0, method='L-BFGS-B', 
                           bounds=[(0.2, 0.95), (0, np.pi/7)], 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                r_opt, th_opt = res.x
                pts_test = _generate_two_ring_points(r_opt, th_opt)
                exact_s = _score(pts_test)
                
                if exact_s > best_score:
                    best_score = exact_s
                    best_r = r_opt
                    best_theta = th_opt
                    best_pts = pts_test
        except Exception:
            pass

    # 3. Fine polish with exact score using small perturbations
    if time.time() < deadline - 5.0:
        # Try a few random perturbations around the best point
        # and use Nelder-Mead on exact score (slower but precise)
        
        def _neg_exact_score(params):
            r, th = params
            if r < 0.1 or r > 0.99:
                return 1e9
            pts = _generate_two_ring_points(r, th)
            return -_score(pts)
            
        # Local Nelder-Mead polish
        try:
            x0 = [best_r, best_theta]
            res = minimize(_neg_exact_score, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 100, 'adaptive': True})
            
            if res.fun < -best_score:
                best_r, best_theta = res.x
                best_pts = _generate_two_ring_points(best_r, best_theta)
                best_score = -res.fun
        except Exception:
            pass

        # Additional random restarts if time permits
        if time.time() < deadline - 2.0:
            for _ in range(10):
                if time.time() > deadline - 2.0:
                    break
                # Perturb
                r_pert = best_r + np.random.normal(0, 0.01)
                th_pert = best_theta + np.random.normal(0, np.pi/100)
                
                if r_pert < 0.2 or r_pert > 0.95:
                    continue
                if th_pert < 0 or th_pert > np.pi/7:
                    continue
                    
                try:
                    res = minimize(_neg_exact_score, [r_pert, th_pert], method='Nelder-Mead', 
                                   options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 50, 'adaptive': True})
                    
                    if res.fun < -best_score:
                        best_r, best_theta = res.x
                        best_pts = _generate_two_ring_points(best_r, best_theta)
                        best_score = -res.fun
                except Exception:
                    pass

    # Final check
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    return best_pts
# EVOLVE-BLOCK-END
