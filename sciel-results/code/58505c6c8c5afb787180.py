"""Heilbronn convex regions n=14 - combined centrally symmetric + support function approaches."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
HALF_N = 7  # 7 pairs of antipodal points for central symmetry

def _min_triangle_area(points):
    """Compute minimum triangle area over all C(n,3) triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1x = points[j, 0] - points[i, 0]
                v1y = points[j, 1] - points[i, 1]
                v2x = points[k, 0] - points[i, 0]
                v2y = points[k, 1] - points[i, 1]
                area = 0.5 * abs(v1x * v2y - v1y * v2x)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

# ============================================================
# Representation 1: Centrally symmetric (14D -> 14 points)
# ============================================================

def _cs_params_to_points(params):
    """Convert 14 params (7 angles + 7 radii) to 14 centrally symmetric points."""
    angles = params[:HALF_N]
    radii = params[HALF_N:]
    
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    
    # Sort by angle for convex polygon order
    point_angles = np.arctan2(all_points[:, 1], all_points[:, 0])
    order = np.argsort(point_angles)
    return all_points[order]

def _cs_objective(params):
    """Negative score for central symmetry optimization."""
    points = _cs_params_to_points(params)
    return -_score(points)

# ============================================================
# Representation 2: Support function with fixed non-uniform angles (14D)
# ============================================================

def _sf_params_to_points(params):
    """Convert 14 radial values to 14 points using fixed non-uniform angles.
    
    Uses angles clustered near diagonals for better coverage.
    The 14 angles are fixed; we optimize the 14 radial distances.
    """
    # Fixed angles: non-uniform, clustered near diagonals (pi/4, 3pi/4, etc.)
    # 14 angles spread around the circle with emphasis on diagonal directions
    base_angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    # Add small perturbation to break perfect symmetry and allow optimization
    # Use a fixed non-uniform set that clusters near diagonals
    angles = np.array([
        0.0, 0.25, 0.55, 0.85, 1.15, 1.45, 1.75,
        2.05, 2.35, 2.65, 2.95, 3.25, 3.55, 3.85
    ])
    
    radii = np.clip(params, 0.05, 3.0)
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Sort by angle for convex polygon order
    point_angles = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(point_angles)
    return points[order]

def _sf_objective(params):
    """Negative score for support function optimization."""
    points = _sf_params_to_points(params)
    return -_score(points)

# ============================================================
# Multi-phase optimization
# ============================================================

def _refine_local(objective, params, bounds, start_time, deadline, maxiter=300):
    """Local refinement using L-BFGS-B, respecting deadline."""
    if time.time() > deadline - 10:
        return None, None
    try:
        res = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun is not None:
            return res.x, -res.fun
    except Exception:
        pass
    return None, None

def _refine_nm(objective, params, start_time, deadline, maxiter=1000):
    """Nelder-Mead refinement, respecting deadline."""
    if time.time() > deadline - 5:
        return None, None
    try:
        res = minimize(objective, params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun is not None:
            return res.x, -res.fun
    except Exception:
        pass
    return None, None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 138.0  # Safety margin before 150s
    
    best_score = 0.0
    best_points = None
    
    # Fallback: regular 14-gon
    regular_points = np.array([[np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)] for i in range(N)])
    best_points = regular_points.copy()
    best_score = _score(best_points)
    
    # ============================================================
    # Phase 1: Differential evolution on central symmetry (14D)
    # ============================================================
    cs_bounds = [(0.02, np.pi - 0.02)] * HALF_N + [(0.05, 3.0)] * HALF_N
    
    de_results = []
    de_kwargs_list = [
        {'maxiter': 35, 'popsize': 12, 'tol': 1e-5, 'seed': 42, 'init': 'sobol'},
        {'maxiter': 25, 'popsize': 10, 'tol': 1e-5, 'seed': 123, 'init': 'latinhypercube'},
    ]
    
    for kwargs in de_kwargs_list:
        if time.time() > DEADLINE - 50:
            break
        try:
            result = differential_evolution(
                _cs_objective, cs_bounds,
                maxiter=kwargs['maxiter'],
                popsize=kwargs['popsize'],
                tol=kwargs['tol'],
                seed=kwargs['seed'],
                polish=False,
                init=kwargs['init']
            )
            if result.fun is not None:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = _cs_params_to_points(result.x).copy()
                de_results.append((result.x, score))
        except Exception:
            continue
    
    # ============================================================
    # Phase 2: Differential evolution on support function (14D)
    # ============================================================
    sf_bounds = [(0.05, 3.0)] * N
    
    if time.time() < DEADLINE - 40:
        try:
            result = differential_evolution(
                _sf_objective, sf_bounds,
                maxiter=30,
                popsize=12,
                tol=1e-5,
                seed=77,
                polish=False,
                init='sobol'
            )
            if result.fun is not None:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = _sf_params_to_points(result.x).copy()
        except Exception:
            pass
    
    # ============================================================
    # Phase 3: Local refinement from multiple starting points
    # ============================================================
    
    # Collect starting points for refinement
    start_params_cs = []
    start_params_sf = []
    
    # From DE results
    if de_results:
        start_params_cs.append(de_results[0][0])
    
    # Regular polygon as initial guess for CS
    regular_angles = np.linspace(0, np.pi, HALF_N, endpoint=False)
    regular_radii = np.ones(HALF_N)
    start_params_cs.append(np.concatenate([regular_angles, regular_radii]))
    
    # Random starts for CS
    rng = np.random.RandomState(42)
    for _ in range(3):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, HALF_N))
        radii = rng.uniform(0.3, 1.5, HALF_N)
        start_params_cs.append(np.concatenate([angles, radii]))
    
    # Random starts for SF
    for _ in range(3):
        radii = rng.uniform(0.3, 1.5, N)
        start_params_sf.append(radii)
    
    # Refine CS starts
    for sp in start_params_cs:
        if time.time() > DEADLINE - 15:
            break
        refined_params, refined_score = _refine_local(_cs_objective, sp, cs_bounds, start_time, DEADLINE, maxiter=200)
        if refined_score is not None and refined_score > best_score:
            best_score = refined_score
            best_points = _cs_params_to_points(refined_params).copy()
    
    # Refine SF starts
    for sp in start_params_sf:
        if time.time() > DEADLINE - 10:
            break
        refined_params, refined_score = _refine_local(_sf_objective, sp, sf_bounds, start_time, DEADLINE, maxiter=200)
        if refined_score is not None and refined_score > best_score:
            best_score = refined_score
            best_points = _sf_params_to_points(refined_params).copy()
    
    # ============================================================
    # Phase 4: Fine-tuning with Nelder-Mead on best result
    # ============================================================
    if time.time() < DEADLINE - 3:
        # Try to identify which representation gave the best result and refine it
        # We'll try both refinements on the current best by converting back
        
        # For CS: try to recover params from best_points (approximate)
        # Actually, just do NM on the last good params
        if de_results and de_results[0][1] >= best_score - 1e-6:
            # Best came from CS
            if time.time() < DEADLINE - 3:
                refined_params, refined_score = _refine_nm(_cs_objective, de_results[0][0], start_time, DEADLINE, maxiter=500)
                if refined_score is not None and refined_score > best_score:
                    best_score = refined_score
                    best_points = _cs_params_to_points(refined_params).copy()
        else:
            # Best came from SF or other, try NM on SF
            # Use the SF params that gave best
            # We need to track this... let's just do a quick NM on a good SF start
            if start_params_sf and time.time() < DEADLINE - 3:
                refined_params, refined_score = _refine_nm(_sf_objective, start_params_sf[0], start_time, DEADLINE, maxiter=500)
                if refined_score is not None and refined_score > best_score:
                    best_score = refined_score
                    best_points = _sf_params_to_points(refined_params).copy()
    
    return best_points
