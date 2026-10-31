import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

# Regular angles
_REGULAR_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _params_to_points(params):
    """
    Convert parameters to points.
    params[0:N]: angular deviations from regular polygon angles
    params[N:2N]: radial deviations (multiplicative factor 1 + dev)
    """
    d_angles = params[:N]
    d_radii = params[N:2*N]
    
    angles = _REGULAR_ANGLES + d_angles
    radii = 1.0 + d_radii
    radii = np.clip(radii, 0.1, 5.0)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Define bounds for DE
    # Angles: +/- pi/13 (half the spacing) to avoid crossing
    angle_bound = np.pi / N
    # Radii: +/- 0.5 (so radii in [0.5, 1.5])
    radius_bound = 0.5
    
    bounds = [(-angle_bound, angle_bound)] * N + [(-radius_bound, radius_bound)] * N
    
    def obj_de(params):
        pts = _params_to_points(params)
        s = _score(pts)
        if s <= 0:
            return 1e10
        return -s
        
    best_points = None
    best_score = -1
    best_params = None
    
    # Stage 1: Differential Evolution
    # DE is slow, so we limit maxiter and popsize
    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(
                obj_de, bounds,
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_res.success or de_res.fun < 1e9:
                best_params = de_res.x
                best_points = _params_to_points(best_params)
                best_score = _score(best_points)
        except Exception:
            pass
            
    # Fallback if DE failed or didn't improve
    if best_score < 0.01:
        best_points = _params_to_points(np.zeros(2*N))
        best_score = _score(best_points)
        best_params = np.zeros(2*N)

    # Stage 2: Polish with L-BFGS-B on soft objective
    if best_points is not None and time.time() < deadline - 20:
        try:
            def obj_soft(x):
                pts = _params_to_points(x)
                return _neg_soft_score(pts, 0.01)
            
            res = minimize(
                obj_soft, best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            cand_params = res.x
            cand_points = _params_to_points(cand_params)
            s = _score(cand_points)
            if s > best_score:
                best_score = s
                best_points = cand_points.copy()
                best_params = cand_params.copy()
        except Exception:
            pass
            
    # Stage 3: Polish with True Objective (Powell)
    if best_points is not None and time.time() < deadline - 10:
        try:
            def obj_true(x):
                pts = _params_to_points(x)
                return _neg_score(pts)
            
            res = minimize(
                obj_true, best_params,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            cand_params = res.x
            cand_points = _params_to_points(cand_params)
            s = _score(cand_points)
            if s > best_score:
                best_score = s
                best_points = cand_points.copy()
                best_params = cand_params.copy()
        except Exception:
            pass

    # Stage 4: Final Nelder-Mead Polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            def obj_nm(x):
                pts = _params_to_points(x)
                return _neg_score(pts)
            
            res = minimize(
                obj_nm, best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_params = res.x
            cand_points = _params_to_points(cand_params)
            s = _score(cand_points)
            if s > best_score:
                best_score = s
                best_points = cand_points.copy()
        except Exception:
            pass

    return best_points
