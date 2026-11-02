"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.
Structural change: centrally symmetric parameterization (7 antipodal pairs) to
reduce the search space from 28D to 14D, combined with a soft-min objective for
smoother gradients and a multi-phase refinement (DE -> L-BFGS-B -> Nelder-Mead).
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(flat):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 points.
    The 14 points are: p_i = r_i * (cos(a_i), sin(a_i)) and -p_i for i=0..6.
    This enforces central symmetry, reducing the dimensionality of the search.
    """
    angles = flat[:N_HALF]
    radii = flat[N_HALF:]
    
    # Ensure angles are in valid range and sorted for convex position
    # We clip to avoid degenerate cases where points overlap or go out of bounds
    angles = np.clip(angles, 0.01, np.pi - 0.01)
    radii = np.clip(radii, 0.01, 5.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective_soft(flat, beta=50.0):
    """
    Soft-min objective for centrally symmetric parameterization.
    Uses a log-sum-exp approximation for the minimum area to provide gradients.
    """
    pts = _params_to_points(flat)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_hard(flat):
    """
    Hard objective for centrally symmetric parameterization.
    Returns negative score for minimization.
    """
    pts = _params_to_points(flat)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_params = None
    best_pts = None

    def update_best(params):
        nonlocal best_score, best_params, best_pts
        if params is None:
            return
        pts = _params_to_points(params)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_params = params.copy()
            best_pts = pts.copy()

    # Bounds for differential evolution
    # Angles: (0.01, pi-0.01), Radii: (0.01, 5.0)
    bounds = [(0.01, np.pi - 0.01)] * N_HALF + [(0.01, 5.0)] * N_HALF

    # Phase 1: Differential Evolution on the reduced 14D space
    # Using a Sobol sequence for better coverage of the reduced space
    if time.time() < deadline - 20:
        try:
            de_res = differential_evolution(
                _objective_soft, bounds,
                seed=42, maxiter=60, tol=1e-8,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun is not None and de_res.fun < 1e9:
                update_best(de_res.x)
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    # Start from the best DE result and some random feasible points
    starts = []
    if best_params is not None:
        starts.append(best_params)
    
    rng = np.random.default_rng(123)
    # Generate diverse random starts in the reduced space
    for _ in range(10):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.5, 2.0, N_HALF)
        starts.append(np.concatenate([angles, radii]))
    
    for i, sp in enumerate(starts):
        if time.time() > deadline - 15:
            break
        try:
            # Use soft objective for smoother gradients in L-BFGS-B
            res = minimize(_objective_soft, sp, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun is not None and res.fun < 1e9:
                update_best(res.x)
        except Exception:
            pass

    # Phase 3: Fine-tuning with Nelder-Mead on the hard objective
    # This allows for non-smooth steps that L-BFGS-B might miss
    if best_params is not None and time.time() < deadline - 5:
        try:
            res = minimize(_objective_hard, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun is not None and -res.fun > best_score:
                update_best(res.x)
        except Exception:
            pass

    # Fallback to regular 14-gon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Note: The regular 14-gon is centrally symmetric, so it fits the parameterization
        # but we just return the points directly as the interface expects points.
    
    # The interface expects an array of shape (n, 2).
    # The score is invariant to scaling, so we don't strictly need to normalize to unit area,
    # but it's good practice to ensure numerical stability.
    # However, the evaluator calculates hull area itself, so returning raw points is fine.
    # The current program normalizes to unit area, so we will do the same to be consistent
    # and avoid any potential issues with very large/small coordinates.
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass

    return best_pts
