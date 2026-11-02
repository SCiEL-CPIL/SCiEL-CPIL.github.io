"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def _score_points(pts):
    """Score: min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min(values, beta):
    """Soft minimum using log-sum-exp (smooth approximation to min)."""
    values = np.asarray(values)
    return -np.log(np.sum(np.exp(-beta * values))) / beta

def _all_triangle_areas(pts):
    """Return array of all triangle areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                areas.append(area)
    return np.array(areas)

def _smooth_score(pts, beta=50.0):
    """Smooth score using soft-min of triangle areas / hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    soft_min_area = _soft_min(areas, beta)
    return soft_min_area / hull_area

def _neg_smooth_score(params, beta=50.0):
    """Negative smooth score for minimization."""
    pts = _params_to_points(params)
    return -_smooth_score(pts, beta)

def _params_to_points(params):
    """Convert parameter vector to point array.
    params: [r_inner, r_outer, angle_inner, angle_outer, scale_x, scale_y]
    Structure: 1 center + 6 inner hexagon + 6 outer hexagon
    """
    r_inner = params[0]
    r_outer = params[1]
    angle_inner = params[2]
    angle_outer = params[3]
    scale_x = params[4]
    scale_y = params[5]
    
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Inner hexagon (6 points)
    for i in range(6):
        theta = angle_inner + i * 2 * np.pi / 6
        x = r_inner * np.cos(theta) * scale_x
        y = r_inner * np.sin(theta) * scale_y
        pts.append([x, y])
    # Outer hexagon (6 points)
    for i in range(6):
        theta = angle_outer + i * 2 * np.pi / 6
        x = r_outer * np.cos(theta) * scale_x
        y = r_outer * np.sin(theta) * scale_y
        pts.append([x, y])
    
    return np.array(pts)

def _params_to_points_v2(params):
    """Alternative: 1 center + 4 inner + 8 outer (square + octagon)."""
    r_inner = params[0]
    r_outer = params[1]
    angle_inner = params[2]
    angle_outer = params[3]
    scale_x = params[4]
    scale_y = params[5]
    
    pts = []
    pts.append([0.0, 0.0])
    for i in range(4):
        theta = angle_inner + i * 2 * np.pi / 4
        x = r_inner * np.cos(theta) * scale_x
        y = r_inner * np.sin(theta) * scale_y
        pts.append([x, y])
    for i in range(8):
        theta = angle_outer + i * 2 * np.pi / 8
        x = r_outer * np.cos(theta) * scale_x
        y = r_outer * np.sin(theta) * scale_y
        pts.append([x, y])
    
    return np.array(pts)

def _params_to_points_v3(params):
    """Alternative: 2 center-ish + 5 inner + 6 outer."""
    r_inner = params[0]
    r_outer = params[1]
    angle_inner = params[2]
    angle_outer = params[3]
    scale_x = params[4]
    scale_y = params[5]
    r_center = params[6] if len(params) > 6 else 0.0
    
    pts = []
    # Two center points
    pts.append([r_center, 0.0])
    pts.append([-r_center, 0.0])
    for i in range(5):
        theta = angle_inner + i * 2 * np.pi / 5
        x = r_inner * np.cos(theta) * scale_x
        y = r_inner * np.sin(theta) * scale_y
        pts.append([x, y])
    for i in range(6):
        theta = angle_outer + i * 2 * np.pi / 6
        x = r_outer * np.cos(theta) * scale_x
        y = r_outer * np.sin(theta) * scale_y
        pts.append([x, y])
    
    return np.array(pts)

def _polish(params, to_points_func, beta_start=20.0, beta_end=200.0, max_iter=50):
    """Polish parameters using smooth min with increasing beta."""
    current_params = np.array(params, dtype=float)
    n_params = len(current_params)
    
    for i in range(max_iter):
        beta = beta_start * (beta_end / beta_start) ** (i / max_iter)
        result = minimize(
            _neg_smooth_score,
            current_params,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < -_neg_smooth_score(current_params, beta):
            current_params = result.x
        else:
            break
    
    # Final evaluation with exact min
    pts = to_points_func(current_params)
    score = _score_points(pts)
    return current_params, score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    # Define symmetry configurations
    configs = [
        ("hex_hex", _params_to_points, [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0.5, 1.5), (0.5, 1.5)]),
        ("sq_oct", _params_to_points_v2, [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0.5, 1.5), (0.5, 1.5)]),
        ("pent_hex", _params_to_points_v3, [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0.5, 1.5), (0.5, 1.5), (0, 0.5)]),
    ]
    
    for config_name, to_points_func, bounds in configs:
        if time.time() > deadline:
            break
        
        # Differential evolution for global search
        try:
            def _de_objective(params):
                pts = to_points_func(params)
                return -_score_points(pts)
            
            result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            
            if time.time() > deadline:
                break
            
            # Polish the best result
            polished_params, score = _polish(result.x, to_points_func)
            
            if score > best_score:
                best_score = score
                best_pts = to_points_func(polished_params)
                
        except Exception as e:
            continue
    
    # If no valid result, use a simple symmetric configuration
    if best_pts is None:
        pts = _params_to_points([0.4, 1.0, 0, 0, 1.0, 1.0])
        best_pts = pts
        best_score = _score_points(pts)
    
    # Final check and normalization
    if best_pts is not None:
        try:
            hull = ConvexHull(best_pts)
            if hull.volume > 1e-12:
                # Normalize to unit area convex hull
                best_pts = best_pts / np.sqrt(hull.volume)
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
