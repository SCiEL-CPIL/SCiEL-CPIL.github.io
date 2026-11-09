import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _get_triangle_indices(n):
    """Precompute indices for all triangles."""
    idxs = list(itertools.combinations(range(n), 3))
    i, j, k = zip(*idxs)
    return np.array(i), np.array(j), np.array(k)

def _compute_areas_from_indices(points, i_idx, j_idx, k_idx):
    """Compute areas using precomputed indices."""
    x1, y1 = points[i_idx, 0], points[i_idx, 1]
    x2, y2 = points[j_idx, 0], points[j_idx, 1]
    x3, y3 = points[k_idx, 0], points[k_idx, 1]
    return 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def _soft_min(areas, alpha):
    """Compute stable soft minimum."""
    min_area = np.min(areas)
    if min_area == 0:
        return 0.0
    shifted = areas - min_area
    exp_vals = np.exp(-alpha * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return min_area
    log_sum_exp = np.log(sum_exp)
    return min_area - (1.0 / alpha) * log_sum_exp

def _objective(x, alpha, i_idx, j_idx, k_idx):
    """
    Objective for L-BFGS-B.
    x: (22,) array for points 2..12.
    """
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2:, :] = x.reshape(11, 2)
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
        
    if A_hull < 1e-6:
        return 1e10
        
    areas = _compute_areas_from_indices(points, i_idx, j_idx, k_idx)
    
    if np.min(areas) < 1e-9:
        return 1e10 - np.min(areas)
        
    A_soft = _soft_min(areas, alpha)
    score = A_soft / A_hull
    return -score

def _true_score(points, i_idx, j_idx, k_idx):
    """Compute the true score A_min / A_hull."""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -np.inf
        
    if A_hull < 1e-6:
        return -np.inf
        
    areas = _compute_areas_from_indices(points, i_idx, j_idx, k_idx)
    A_min = np.min(areas)
    if A_min < 1e-9:
        return -np.inf
        
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    max_time = 600.0
    n = 13
    
    # Precompute indices
    i_idx, j_idx, k_idx = _get_triangle_indices(n)
    
    # Helper to transform points to fixed system P0=(0,0), P1=(1,0)
    def _transform_to_fixed_system(points):
        p0 = points[0]
        p1 = points[1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return points
        s = 1.0 / norm_v
        c = v[0] / norm_v
        s_ = v[1] / norm_v
        R = np.array([[c, -s_], [s_, c]])
        transformed = (points - p0) @ R.T * s
        return transformed

    # Initializations
    inits = []
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(_transform_to_fixed_system(p_reg))
    
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        p_rand = rng.uniform(-1, 1, size=(n, 2))
        # Check rank
        if np.linalg.matrix_rank(np.column_stack([p_rand[:,0]-p_rand[0,0], p_rand[:,1]-p_rand[0,1]])) < 2:
            continue
        inits.append(_transform_to_fixed_system(p_rand))
        
    p_pert = p_reg + rng.normal(0, 0.1, size=(n, 2))
    inits.append(_transform_to_fixed_system(p_pert))

    best_score = -np.inf
    best_points = None
    best_x = None
    
    alphas = [5, 10, 20, 50, 100, 200, 500]
    bounds = [(-10, 10)] * 22
    
    for alpha in alphas:
        if time.time() - start_time > max_time - 60:
            break
            
        starts = []
        if best_x is not None:
            starts.append(best_x)
        starts.append(inits[0][2:].flatten()) # Regular polygon init
        starts.append(rng.uniform(-2, 2, size=22))
        
        for start_x in starts:
            if time.time() - start_time > max_time - 30:
                break
            try:
                res = minimize(
                    _objective, 
                    start_x, 
                    args=(alpha, i_idx, j_idx, k_idx), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                x_opt = res.x
                points_opt = np.zeros((13, 2))
                points_opt[0] = [0.0, 0.0]
                points_opt[1] = [1.0, 0.0]
                points_opt[2:, :] = x_opt.reshape(11, 2)
                
                score = _true_score(points_opt, i_idx, j_idx, k_idx)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt
                    best_x = x_opt
                    
            except Exception:
                pass
                
    # Final polish with Nelder-Mead if time allows
    if best_points is not None and time.time() - start_time < max_time - 10:
        def _nm_objective(x):
            points = np.zeros((13, 2))
            points[0] = [0.0, 0.0]
            points[1] = [1.0, 0.0]
            points[2:, :] = x.reshape(11, 2)
            score = _true_score(points, i_idx, j_idx, k_idx)
            return -score if score != -np.inf else 1e10
            
        try:
            res_nm = minimize(
                _nm_objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            x_nm = res_nm.x
            points_nm = np.zeros((13, 2))
            points_nm[0] = [0.0, 0.0]
            points_nm[1] = [1.0, 0.0]
            points_nm[2:, :] = x_nm.reshape(11, 2)
            
            score_nm = _true_score(points_nm, i_idx, j_idx, k_idx)
            if score_nm > best_score:
                best_score = score_nm
                best_points = points_nm
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = inits[0]
        
    return best_points
