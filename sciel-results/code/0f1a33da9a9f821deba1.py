import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # Safety margin against 120s limit

# Precompute all triple indices for vectorized area calculation
_ALL_TRIPLES = np.array(list(combinations(range(N), 3)))

def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    p1 = points[_ALL_TRIPLES[:, 0]]
    p2 = points[_ALL_TRIPLES[:, 1]]
    p3 = points[_ALL_TRIPLES[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    if points.shape != (N, 2):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_triangle_area(points, beta=200.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    soft_min(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    """
    areas = _all_triangle_areas(points)
    # Clip to avoid overflow/underflow issues
    areas = np.clip(areas, 1e-15, None)
    neg_beta_a = -beta * areas
    # Shift for numerical stability
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    soft_min = -log_sum / beta
    return float(soft_min)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 1.0

def _soft_objective(x, beta=200.0):
    """
    Smooth objective: -soft_min_triangle_area / hull_area.
    """
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    sm = _soft_min_triangle_area(pts, beta=beta)
    if sm < 1e-15:
        return 1e10
    return -sm / hull_area

def _exact_objective(x):
    """Exact nonsmooth objective for final polish."""
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    min_area = np.min(_all_triangle_areas(pts))
    return -min_area / hull_area

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    # 1. Generate diverse initial configurations
    # Strategy: Mix of symmetric topologies (12-gon+center, 13-gon, etc.)
    
    configs = []
    
    # Config 1: 12-gon + center (Current best topology)
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([pts_12, center]))
    
    # Config 2: Regular 13-gon
    angles_13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts_13 = np.column_stack((np.cos(angles_13), np.sin(angles_13)))
    configs.append(pts_13)
    
    # Config 3: 11-gon + 2 inner points (opposite)
    angles_11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    pts_11 = np.column_stack((np.cos(angles_11), np.sin(angles_11)))
    inner_1 = np.array([[0.5, 0.0]])
    inner_2 = np.array([[-0.5, 0.0]])
    configs.append(np.vstack([pts_11, inner_1, inner_2]))
    
    # Config 4: Jittered Grid
    rng = np.random.default_rng(42)
    grid = np.array([(i, j) for i in range(4) for j in range(4)])[:13]
    grid = grid + rng.normal(0, 0.1, grid.shape)
    configs.append(grid)
    
    # Config 5: Random jittered 12-gon + center
    angles_rand = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.normal(0, 0.2, 12)
    pts_rand = np.column_stack((np.cos(angles_rand), np.sin(angles_rand)))
    c_rand = rng.normal(0, 0.1, 2)
    configs.append(np.vstack([pts_rand, c_rand.reshape(1,2)]))

    # 2. Optimization Loop
    # Phase 1: Smooth optimization with soft-min to find good basins
    # Phase 2: Exact optimization to polish
    
    for i_init, init_pts in enumerate(configs):
        if _time_remaining() < 15.0:
            break
            
        x0 = init_pts.flatten()
        
        # Step 1: Smooth Optimization (Nelder-Mead on soft objective)
        # Use adaptive Nelder-Mead with tight tolerances
        try:
            res_soft = minimize(
                _soft_objective, 
                x0, 
                args=(150.0,), # beta
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            x_soft = res_soft.x
        except Exception:
            x_soft = x0

        if _time_remaining() < 10.0:
            break

        # Step 2: Exact Optimization (Nelder-Mead on exact objective)
        # Start from the result of the soft optimization
        try:
            res_exact = minimize(
                _exact_objective, 
                x_soft, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            x_final = res_exact.x
        except Exception:
            x_final = x_soft

        # Evaluate
        candidate_points = x_final.reshape(N, 2)
        score = get_score(candidate_points)
        
        if score > best_score:
            best_score = score
            best_points = candidate_points.copy()
            
        # If we have time, try a few more random perturbations of the best solution found so far
        # This acts as a local search around the best basin
        if _time_remaining() > 20.0 and best_points is not None:
             # Try 2-3 local refinements
             for _ in range(2):
                 if _time_remaining() < 10.0:
                     break
                 # Perturb best
                 noise = np.random.normal(0, 0.05, 2*N)
                 x_perturb = best_points.flatten() + noise
                 
                 try:
                     res_p = minimize(
                         _exact_objective, 
                         x_perturb, 
                         method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                     )
                     x_p_final = res_p.x
                     cand = x_p_final.reshape(N, 2)
                     s = get_score(cand)
                     if s > best_score:
                         best_score = s
                         best_points = cand.copy()
                 except Exception:
                     pass

    if best_points is None:
        # Fallback to the first config
        best_points = configs[0]
        
    return best_points
