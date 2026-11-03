import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _compute_radius(C):
    """Compute the radius for a given set of centers."""
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    if r_bound < 0:
        return -r_bound + 10.0
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    return min(r_bound, r_pair)

def _soft_min(a, b, beta=50.0):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    m = np.minimum(a, b)
    return m - np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta

def construct():
    V = _generate_bcc_points(4)
    
    # Compute extent of V
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    V_extent = V_max - V_min
    max_extent = np.max(V_extent)
    
    # Better initial scale: target radius ~0.111
    # In BCC, min neighbor distance = sqrt(3)/2 * s (where s is the scale)
    # r_pair = sqrt(3)/4 * s
    # r_bound ≈ (1 - s * max_extent) / 2
    # Equate: sqrt(3)/4 * s = (1 - s * max_extent) / 2
    # sqrt(3)/2 * s = 1 - s * max_extent
    # s * (sqrt(3)/2 + max_extent) = 1
    s_init = 1.0 / (np.sqrt(3.0) / 2.0 + max_extent)
    
    # Adjust to target radius
    target_r = 0.1113823475125
    # r_pair = sqrt(3)/4 * s, so s = 4 * r / sqrt(3)
    s_target = 4.0 * target_r / np.sqrt(3.0)
    s_init = max(s_init, s_target * 0.95)  # Start slightly below target
    
    x0 = np.concatenate(([s_init], [0.0, 0.0, 0.0]))
    
    best_result = None
    best_score = -np.inf
    
    def objective(params, beta=50.0):
        s = params[0]
        t = params[1:4]
        C = s * V + t
        
        dist_to_bound = 0.5 - np.abs(C).max(axis=1)
        r_bound = np.min(dist_to_bound)
        
        if r_bound < -0.01:
            return 10.0 - r_bound
        
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        r_pair = min_dist / 2.0
        
        # Use soft-min for smoothness
        r = _soft_min(r_bound, r_pair, beta)
        return -r
    
    def objective_exact(params):
        s = params[0]
        t = params[1:4]
        C = s * V + t
        
        dist_to_bound = 0.5 - np.abs(C).max(axis=1)
        r_bound = np.min(dist_to_bound)
        
        if r_bound < -0.01:
            return 10.0 - r_bound
        
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        r_pair = min_dist / 2.0
        
        r = min(r_bound, r_pair)
        return -r
    
    # Phase 1: Coarse optimization with soft-min
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
        
        s_opt = res.x[0]
        t_opt = res.x[1:4]
        C = s_opt * V + t_opt
        score = _compute_radius(C)
        
        if score > best_score and score > 0.05:
            best_score = score
            best_result = C.copy()
            
    except Exception:
        pass
    
    # Phase 2: Polish with exact objective and tighter tolerances
    if best_result is not None and not _check_time():
        try:
            x_polish = np.concatenate(([best_result[0][0] / V[0][0] if V[0][0] != 0 else s_init], 
                                       [best_result[:, 0].mean() - s_init * V[:, 0].mean(),
                                        best_result[:, 1].mean() - s_init * V[:, 1].mean(),
                                        best_result[:, 2].mean() - s_init * V[:, 2].mean()]))
            # Recompute s and t properly
            # C = s * V + t => t = C - s * V
            # Use the best s from phase 1
            s_best = res.x[0]
            t_best = res.x[1:4]
            x_polish = np.concatenate(([s_best], t_best))
            
            res2 = minimize(objective_exact, x_polish, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            
            s_opt2 = res2.x[0]
            t_opt2 = res2.x[1:4]
            C2 = s_opt2 * V + t_opt2
            score2 = _compute_radius(C2)
            
            if score2 > best_score and score2 > 0.05:
                best_score = score2
                best_result = C2.copy()
                
        except Exception:
            pass
    
    # Phase 3: Multi-start with diverse perturbations
    if not _check_time():
        np.random.seed(42)
        for i in range(5):
            if _check_time():
                break
            # Diverse perturbations
            pert_scale = np.random.uniform(0.85, 1.15)
            pert_trans = np.random.uniform(-0.02, 0.02, 3)
            x0_pert = np.concatenate(([s_init * pert_scale], pert_trans))
            
            try:
                res_pert = minimize(objective, x0_pert, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                
                s_pert = res_pert.x[0]
                t_pert = res_pert.x[1:4]
                C_pert = s_pert * V + t_pert
                score_pert = _compute_radius(C_pert)
                
                if score_pert > best_score and score_pert > 0.05:
                    best_score = score_pert
                    best_result = C_pert.copy()
                    
                    # Polish the best found
                    if not _check_time():
                        try:
                            x_polish2 = np.concatenate(([s_pert], t_pert))
                            res_polish = minimize(objective_exact, x_polish2, method='L-BFGS-B',
                                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                            s_pol = res_polish.x[0]
                            t_pol = res_polish.x[1:4]
                            C_pol = s_pol * V + t_pol
                            score_pol = _compute_radius(C_pol)
                            if score_pol > best_score:
                                best_score = score_pol
                                best_result = C_pol.copy()
                        except Exception:
                            pass
                            
            except Exception:
                pass
    
    if best_result is not None and best_score > 0.05:
        return best_result
    
    # Fallback to seed grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]
