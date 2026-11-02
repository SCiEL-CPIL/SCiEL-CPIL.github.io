"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _smooth_neg_score(x, beta):
    """
    Negative smooth surrogate for the score.
    We want to maximize min(A_i) / H.
    Equivalent to maximizing log(min(A_i)) - log(H).
    
    Using log-sum-exp for the min:
    log(min(a_i)) = -1/beta * log(sum(exp(-beta * a_i)))
    
    So objective to minimize:
    f(x) = 1/beta * log(sum(exp(-beta * A_i))) + log(H)
    
    Note: A_i and H are positive.
    """
    pts = x.reshape(N, 2)
    
    # Compute areas
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Clip areas to avoid log(0) or negative values due to numerical issues
    areas = np.maximum(areas, 1e-12)
    
    # Log-sum-exp trick for stability
    # We need sum(exp(-beta * areas))
    # Let m = min(-beta * areas) = -beta * max(areas)
    # sum(exp(-beta * areas)) = exp(m) * sum(exp(-beta * areas - m))
    # log(sum(...)) = m + log(sum(exp(-beta * areas - m)))
    
    neg_beta_areas = -beta * areas
    m = np.max(neg_beta_areas)
    if not np.isfinite(m):
        return 1e10
        
    log_sum_exp = m + np.log(np.sum(np.exp(neg_beta_areas - m)))
    
    # Compute Hull Area
    try:
        hull_area = _convex_hull_area(pts)
        if hull_area <= 1e-12:
            return 1e10
        log_hull = np.log(hull_area)
    except Exception:
        return 1e10

    # Objective: 1/beta * log_sum_exp + log_hull
    # We want to minimize this.
    obj = (1.0 / beta) * log_sum_exp + log_hull
    
    return obj

def _generate_initial_points():
    """
    Generate a diverse set of initial points.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts_reg)
    
    # 2. Regular 12-gon + Center
    angles_12 = 2 * np.pi * np.arange(12) / 12
    pts_12 = np.column_stack([0.5 + 0.4 * np.cos(angles_12), 0.5 + 0.4 * np.sin(angles_12)])
    pts_12_c = np.vstack([pts_12, np.array([0.5, 0.5])])
    inits.append(pts_12_c)
    
    # 3. Jittered Grid (4x4 minus 3 random points)
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    rng = np.random.RandomState(123)
    indices_to_keep = list(range(16))
    removed = rng.choice(16, 3, replace=False)
    indices_to_keep = [i for i in indices_to_keep if i not in removed]
    pts_grid = grid_pts[indices_to_keep]
    inits.append(pts_grid)
    
    # 4. Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts_rand = rng.rand(N, 2) * 0.8 + 0.1
        inits.append(pts_rand)
        
    # 5. More structured: Two concentric regular polygons
    # 7 outer, 6 inner
    angles_out = 2 * np.pi * np.arange(7) / 7
    angles_in = 2 * np.pi * np.arange(6) / 6 + np.pi/6 # offset
    pts_out = np.column_stack([0.5 + 0.45 * np.cos(angles_out), 0.5 + 0.45 * np.sin(angles_out)])
    pts_in = np.column_stack([0.5 + 0.2 * np.cos(angles_in), 0.5 + 0.2 * np.sin(angles_in)])
    pts_two_ring = np.vstack([pts_out, pts_in])
    inits.append(pts_two_ring)
    
    # 6. 5 outer, 5 middle, 3 inner
    angles_5 = 2 * np.pi * np.arange(5) / 5
    angles_5b = 2 * np.pi * np.arange(5) / 5 + np.pi/5
    angles_3 = 2 * np.pi * np.arange(3) / 3
    pts_5a = np.column_stack([0.5 + 0.45 * np.cos(angles_5), 0.5 + 0.45 * np.sin(angles_5)])
    pts_5b = np.column_stack([0.5 + 0.25 * np.cos(angles_5b), 0.5 + 0.25 * np.sin(angles_5b)])
    pts_3 = np.column_stack([0.5 + 0.1 * np.cos(angles_3), 0.5 + 0.1 * np.sin(angles_3)])
    pts_three_ring = np.vstack([pts_5a, pts_5b, pts_3])
    inits.append(pts_three_ring)

    return inits

def _local_polish(x0, max_iter=5000):
    """
    Perform a local search using Nelder-Mead on the EXACT objective to refine the starting point x0.
    """
    def _neg_exact(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    try:
        res = minimize(_neg_exact, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
        return res.x
    except Exception:
        return x0

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])

    best_x = None
    best_score = -np.inf
    
    inits = _generate_initial_points()
    
    # Evaluate initial points
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_x = init.flatten()

    # Multi-stage optimization with increasing beta
    # Start with low beta (smooth, broad) and increase to high beta (sharp, accurate)
    # This helps escape local minima of the sharp min() function.
    
    # We will run L-BFGS-B from the best initial point found so far.
    # We might also run from other good inits if time permits.
    
    # Beta schedule: 5, 10, 20, 50, 100, 200
    # Each stage: minimize _smooth_neg_score with current beta
    # Then use the result as starting point for next stage.
    
    # To improve chance of global optimum, we can run this pipeline from top 3 initial candidates.
    
    # Sort inits by score
    init_scores = [( _score(i), i) for i in inits]
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 3 distinct starting points
    candidates = [init_scores[0][1]]
    if len(init_scores) > 1 and init_scores[1][0] > init_scores[0][0] * 0.9:
        candidates.append(init_scores[1][1])
    if len(init_scores) > 2 and init_scores[2][0] > init_scores[0][0] * 0.8:
        candidates.append(init_scores[2][1])
        
    # If best_x is not in candidates, add it (it should be, as it's the best)
    # Actually best_x is the best of all inits, so it's likely in candidates.
    # Let's just use the top candidates from the sorted list.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for cand in candidates:
        if time.time() - _START > _DEADLINE - _SAFETY - 5.0:
            break
            
        current_x = cand.flatten()
        
        # Run multi-stage optimization
        for beta in betas:
            if time.time() - _START > _DEADLINE - _SAFETY - 2.0:
                break
                
            try:
                res = minimize(
                    _smooth_neg_score, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
        
        # Evaluate the result of the smooth optimization
        s = _score(current_x.reshape(N, 2))
        if s > best_score:
            best_score = s
            best_x = current_x

    # Final local polish on the best result found using the EXACT objective
    if time.time() - _START < _DEADLINE - _SAFETY - 2.0 and best_x is not None:
        polished_x = _local_polish(best_x, max_iter=3000)
        polished_score = _score(polished_x.reshape(N, 2))
        
        if polished_score > best_score:
            best_score = polished_score
            best_x = polished_x

    # If we still have time, do one more polish pass with tighter tolerance
    if time.time() - _START < _DEADLINE - _SAFETY - 1.0 and best_x is not None:
        polished_x2 = _local_polish(best_x, max_iter=2000)
        polished_score2 = _score(polished_x2.reshape(N, 2))
        
        if polished_score2 > best_score:
            best_score = polished_score2
            best_x = polished_x2

    return best_x.reshape(N, 2)
