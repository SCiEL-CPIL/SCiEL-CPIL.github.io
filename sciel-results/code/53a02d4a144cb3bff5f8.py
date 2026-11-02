import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute triangle indices for vectorization
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas vectorized
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def smooth_neg_score(x, tau=0.01):
    """
    Objective function using soft-min to provide a differentiable landscape.
    Returns negative score (for minimization).
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e9
    except Exception:
        return 1e9
        
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
    
    # Soft min calculation
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def neg_score(x):
    """
    Exact objective function for minimization.
    """
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def _expand_symmetric(params):
    """
    Expand 7 pairs of points (14D) to 14 points (28D) with point symmetry.
    params: (7, 2) array.
    """
    # params is (7, 2)
    # Points: p1, -p1, p2, -p2, ...
    # We interleave them or just stack. Order doesn't matter for score.
    pos = params
    neg = -params
    # Stack: (p1, -p1, p2, -p2...)
    # Reshape to (7, 2, 2) then reshape to (14, 2)
    pairs = np.stack([pos, neg], axis=1) # (7, 2, 2)
    return pairs.reshape(14, 2)

def _smooth_neg_score_symmetric(params, tau=0.01):
    """
    Objective for symmetric 14D space.
    """
    points = _expand_symmetric(params)
    return smooth_neg_score(points.flatten(), tau)

def _neg_score_symmetric(params):
    """
    Exact objective for symmetric 14D space.
    """
    points = _expand_symmetric(params)
    return neg_score(points.flatten())

def local_refine(points, start_time, deadline):
    """
    Refine a point configuration using L-BFGS-B on smooth objective, then Nelder-Mead on exact.
    """
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    x0 = best_points.flatten()
    
    # Phase 1: L-BFGS-B with annealing tau
    taus = [0.1, 0.05, 0.01, 0.005]
    current_x = x0
    
    for tau in taus:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x: smooth_neg_score(x, tau=tau),
                current_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            # Check if improvement
            if res.fun < smooth_neg_score(current_x, tau=tau):
                current_x = res.x
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on exact score
    if time.time() < deadline - 2:
        try:
            res_nm = minimize(
                neg_score,
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            points_nm = res_nm.x.reshape((N, 2))
            s_nm = compute_score(points_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_points = points_nm
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if not np.all(np.isfinite(points)):
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Generate diverse initial points
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. 7+7 Rings (Optimized radii roughly)
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a_in = a_out + np.pi/7
    p_77 = np.vstack((
        np.column_stack((np.cos(a_out), np.sin(a_out))),
        np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
    ))
    
    # 3. Point Symmetric Random (7 pairs)
    rng = np.random.default_rng(42)
    pairs = rng.normal(0, 1, (7, 2))
    p_sym = _expand_symmetric(pairs)
    
    # 4. Point Symmetric Regular (7 pairs on circle)
    angles_sym = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pairs_reg = np.column_stack((np.cos(angles_sym), np.sin(angles_sym)))
    p_sym_reg = _expand_symmetric(pairs_reg)
    
    starts = [p_reg, p_77, p_sym, p_sym_reg]
    
    # Refine each start
    for start in starts:
        if time.time() > deadline - 10:
            break
        refined_pts, s = local_refine(start, start_time, deadline)
        try_update(refined_pts)
        
    # If time permits, run DE on Symmetric 14D space
    if time.time() < deadline - 20:
        # Bounds for 7 pairs (14D)
        # x in [-2, 2], y in [-2, 2]
        bounds = [(-2, 2)] * 14
        try:
            res_de = differential_evolution(
                _neg_score_symmetric,
                bounds,
                seed=123,
                maxiter=50,
                tol=1e-6,
                polish=True,
                init='sobol',
                popsize=10,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if res_de.fun < 0:
                pts_de = _expand_symmetric(res_de.x.reshape(7, 2))
                # Refine the DE result
                if time.time() < deadline - 5:
                    refined_pts, s = local_refine(pts_de, start_time, deadline)
                    try_update(refined_pts)
        except Exception:
            pass
            
    # If time still permits, try DE on Full 28D space with small population
    if time.time() < deadline - 30:
        bounds_full = [(-2, 2)] * 28
        try:
            # Use a smaller population for 28D
            res_de_full = differential_evolution(
                neg_score,
                bounds_full,
                seed=999,
                maxiter=20,
                tol=1e-4,
                polish=False, # We will polish manually
                init='sobol',
                popsize=5,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if res_de_full.fun < 0:
                pts_de_full = res_de_full.x.reshape(N, 2)
                if time.time() < deadline - 5:
                    refined_pts, s = local_refine(pts_de_full, start_time, deadline)
                    try_update(refined_pts)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = p_reg
        
    return best_points
