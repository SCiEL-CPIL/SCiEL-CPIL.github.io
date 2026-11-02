"""Centrally symmetric polygon optimization for Heilbronn n=14."""
N = 14
M = N // 2  # 7 unique points

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def _build_symmetric_points(unique_pts):
    """Build 14 points from 7 unique points and their negatives."""
    neg_pts = -unique_pts
    # Interleave or just stack? 
    # For convex hull area and triangle areas, the set of points is what matters.
    # The order doesn't matter for the score calculation (min over all triples, hull area).
    # So we can just stack them.
    return np.vstack([unique_pts, neg_pts])

def _min_tri_area(pts):
    """Compute minimum triangle area for N points."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    min_a = float('inf')
    # Precompute all pairs? No, N=14 is small.
    # Triple loop is O(N^3) = 2744. Very fast.
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i+1, n):
            xj, yj = x[j], y[j]
            dx_j = xj - xi
            dy_j = yj - yi
            for k in range(j+1, n):
                xk, yk = x[k], y[k]
                # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
                # Or cross product of (Pj-Pi) and (Pk-Pi)
                dx_k = xk - xi
                dy_k = yk - yi
                cross = dx_j * dy_k - dy_j * dx_k
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def _score_points(pts):
    """Min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    min_a = _min_tri_area(pts)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

def _objective_sym(params):
    """Objective for symmetric optimization. params shape (M, 2)."""
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return 1e10
    unique_pts = params.reshape(M, 2)
    # Ensure no points are at origin (degenerate)
    if np.any(np.linalg.norm(unique_pts, axis=1) < 1e-6):
        return 1e10
    pts = _build_symmetric_points(unique_pts)
    s = _score_points(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _refine_nm(params, deadline):
    """Nelder-Mead refinement in (M, 2) space."""
    x0 = params.flatten().copy()
    
    def obj(flat):
        if time.time() > deadline:
            return 1e10
        p = flat.reshape(M, 2)
        return _objective_sym(p)
        
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < 1e9:
            return res.x.reshape(M, 2), -res.fun
    except Exception:
        pass
    return params, _score_points(_build_symmetric_points(params))

def construct():
    """Place 14 points in convex position to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_params = None
    best_score = 0.0
    best_pts = None
    
    # Initial guess: Regular 14-gon first 7 vertices
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    init_params = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 1. Differential Evolution for global search
    # Bounds: [-2, 2] for x and y
    bounds = [(-2.0, 2.0)] * (2 * M)
    
    try:
        # Use a moderate number of iterations to stay within time
        # DE is stochastic, so set seed for reproducibility? No, let it explore.
        # popsize=20, maxiter=200 might be too slow if objective is expensive?
        # Objective is ~1ms. 20*14*200 = 56000 evals * 1ms = 56s. That's risky.
        # Let's reduce. popsize=15, maxiter=100. 15*14*100 = 21000 evals * 1ms = 21s.
        # We have 140s budget.
        # Let's try popsize=20, maxiter=50. 20*14*50 = 14000 evals. ~14s.
        
        res_de = differential_evolution(
            _objective_sym, bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20,
            tol=1e-6,
            init='latinhypercube',
            polish=False  # We will polish manually
        )
        
        if res_de.fun < 1e9:
            best_params = res_de.x.reshape(M, 2)
            best_score = -res_de.fun
            best_pts = _build_symmetric_points(best_params)
            
    except Exception:
        pass
    
    # 2. Refine with Nelder-Mead
    if best_params is None:
        best_params = init_params
        best_score = _score_points(_build_symmetric_points(best_params))
        best_pts = _build_symmetric_points(best_params)
        
    if time.time() < deadline - 20:
        refined_params, s_ref = _refine_nm(best_params, deadline)
        if s_ref > best_score:
            best_score = s_ref
            best_params = refined_params
            best_pts = _build_symmetric_points(refined_params)
            
    # 3. Multi-start local refinement from random perturbations of best
    rng = np.random.RandomState(123)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        # Perturb current best
        noise = rng.normal(0, 0.1, (M, 2))
        perturbed = best_params + noise
        p_ref, s_pert = _refine_nm(perturbed, deadline)
        if s_pert > best_score:
            best_score = s_pert
            best_params = p_ref
            best_pts = _build_symmetric_points(p_ref)

    # 4. Final Polish with Powell on the full 28D space? 
    # Symmetry might be slightly broken by numerical noise or local minima?
    # But the problem structure suggests symmetry is optimal.
    # Let's stick to symmetric refinement to be safe and fast.
    
    if best_pts is None:
        best_pts = _build_symmetric_points(init_params)
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
        if hull_area > 1e-12:
            scale = np.sqrt(1.0 / hull_area)
            best_pts = best_pts * scale
        return best_pts

    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(best_pts)
        hull_area = hull.volume

    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_pts = best_pts * scale

    return best_pts
# EVOLVE-BLOCK-END
