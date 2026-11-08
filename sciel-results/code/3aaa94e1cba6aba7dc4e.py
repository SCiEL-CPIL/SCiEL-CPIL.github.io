"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area_vectorized(points):
    """Compute the minimum area over all triples of points using vectorization."""
    n = len(points)
    # Create index arrays for all triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only keep i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Get coordinates
    ai = points[i_idx]
    aj = points[j_idx]
    ak = points[k_idx]
    
    # Compute cross product: (bj - bi) x (bk - bi)
    # = (aj - ai) x (ak - ai)
    v1 = aj - ai
    v2 = ak - ai
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    hull = ConvexHull(points)
    return hull.volume  # In 2D, volume is area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    try:
        min_area = _min_triangle_area_vectorized(points)
        hull_area = _convex_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min_triangle_area(points, beta=100.0):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp."""
    n = len(points)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    ai = points[i_idx]
    aj = points[j_idx]
    ak = points[k_idx]
    
    v1 = aj - ai
    v2 = ak - ai
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log-sum-exp(-beta * areas) / beta
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp / beta
    
    return soft_min

def _soft_score(points, beta=100.0):
    """Compute a smooth score using soft-min for the triangle areas."""
    try:
        min_area_soft = _soft_min_triangle_area(points, beta)
        hull_area = _convex_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area_soft / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Start with regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = points.copy()
    best_score = _score(best_points)
    
    from scipy.optimize import minimize
    
    # Phase 1: Nelder-Mead with exact objective
    def objective_exact(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    x0 = best_points.flatten()
    best_x = x0.copy()
    best_obj = objective_exact(best_x)
    
    def callback_nm(xk):
        nonlocal best_x, best_obj
        current_obj = objective_exact(xk)
        if current_obj < best_obj:
            best_obj = current_obj
            best_x = xk.copy()
        return time.time() > deadline - 5
    
    try:
        result = minimize(objective_exact, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                 'adaptive': True},
                         callback=callback_nm)
        obj_val = objective_exact(result.x)
        if obj_val < best_obj:
            best_obj = obj_val
            best_x = result.x.copy()
    except Exception:
        pass
    
    best_points = best_x.reshape(N, 2)
    best_score = _score(best_points)
    
    # Phase 2: Polish with soft objective (smoother landscape)
    remaining = deadline - time.time()
    if remaining > 20:
        def objective_soft(x):
            pts = x.reshape(N, 2)
            return -_soft_score(pts, beta=200.0)
        
        x0_soft = best_points.flatten()
        best_x_soft = x0_soft.copy()
        best_obj_soft = objective_soft(best_x_soft)
        
        def callback_soft(xk):
            nonlocal best_x_soft, best_obj_soft
            current_obj = objective_soft(xk)
            if current_obj < best_obj_soft:
                best_obj_soft = current_obj
                best_x_soft = xk.copy()
            return time.time() > deadline - 5
        
        try:
            result = minimize(objective_soft, x0_soft, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                     'adaptive': True},
                             callback=callback_soft)
            obj_val = objective_soft(result.x)
            if obj_val < best_obj_soft:
                best_obj_soft = obj_val
                best_x_soft = result.x.copy()
        except Exception:
            pass
        
        # Evaluate the soft-optimized point with exact score
        candidate = best_x_soft.reshape(N, 2)
        sc = _score(candidate)
        if sc > best_score:
            best_score = sc
            best_points = candidate.copy()
    
    # Phase 3: Final polish with very tight tolerance
    remaining = deadline - time.time()
    if remaining > 10:
        def objective_tight(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        x0_tight = best_points.flatten()
        best_x_tight = x0_tight.copy()
        best_obj_tight = objective_tight(best_x_tight)
        
        def callback_tight(xk):
            nonlocal best_x_tight, best_obj_tight
            current_obj = objective_tight(xk)
            if current_obj < best_obj_tight:
                best_obj_tight = current_obj
                best_x_tight = xk.copy()
            return time.time() > deadline - 3
        
        try:
            result = minimize(objective_tight, x0_tight, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16,
                                     'adaptive': True},
                             callback=callback_tight)
            obj_val = objective_tight(result.x)
            if obj_val < best_obj_tight:
                best_obj_tight = obj_val
                best_x_tight = result.x.copy()
        except Exception:
            pass
        
        candidate = best_x_tight.reshape(N, 2)
        sc = _score(candidate)
        if sc > best_score:
            best_score = sc
            best_points = candidate.copy()
    
    return best_points
# EVOLVE-BLOCK-END
