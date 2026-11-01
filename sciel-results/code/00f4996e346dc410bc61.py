import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_params(params):
    """
    Generate points from parameters for a 7+7 ring structure.
    params: [r_ratio, rot_inner, jitter_0, ..., jitter_13]
    r_ratio: radius of inner ring relative to outer (fixed at 1.0 for outer)
    rot_inner: rotation offset for inner ring
    jitters: angular perturbations for each point (7 for outer, 7 for inner)
    """
    r_outer = 1.0
    r_inner = params[0]
    rot_inner = params[1]
    jitters = params[2:]
    
    # Outer ring: 7 points
    base_angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_outer = base_angles_outer + jitters[:7]
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    
    # Inner ring: 7 points
    base_angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rot_inner
    angles_inner = base_angles_inner + jitters[7:]
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([pts_outer, pts_inner])

def _neg_score_params(params):
    """
    Objective function for parameter optimization.
    Returns negative score.
    """
    try:
        pts = _generate_two_ring_params(params)
        # Ensure points are not degenerate
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e10
        sc = _score(pts)
        return -sc
    except Exception:
        return 1e10

def _generate_one_center_params(params):
    """
    Generate points from parameters for a 1 center + 13 outer ring structure.
    params: [r_outer, rot_outer, jitter_0, ..., jitter_12]
    """
    r_outer = params[0]
    rot_outer = params[1]
    jitters = params[2:]
    
    # Center point
    pt_center = np.array([[0.0, 0.0]])
    
    # Outer ring: 13 points
    base_angles = np.linspace(0, 2 * np.pi, 13, endpoint=False) + rot_outer
    angles = base_angles + jitters
    pts_outer = np.column_stack([r_outer * np.cos(angles), r_outer * np.sin(angles)])
    
    return np.vstack([pt_center, pts_outer])

def _neg_score_one_center(params):
    """
    Objective function for 1+13 parameter optimization.
    """
    try:
        pts = _generate_one_center_params(params)
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e10
        sc = _score(pts)
        return -sc
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Initialize with regular 14-gon
    reg_pts = _regular_polygon(14, 1.0, 0.0)
    _update_best(reg_pts, _score(reg_pts))
    
    rng = np.random.RandomState(42)

    # Strategy 1: Optimize 7+7 ring structure
    # Initial guess: r_inner=0.7, rot=pi/7 (staggered), jitters=0
    init_params_77 = np.array([0.7, np.pi/7.0] + [0.0]*14)
    
    if _time_left() > 10:
        try:
            # Use L-BFGS-B for smooth optimization
            res = scipy.optimize.minimize(
                _neg_score_params,
                init_params_77,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            best_params_77 = res.x
            best_pts_77 = _generate_two_ring_params(best_params_77)
            _update_best(best_pts_77, _score(best_pts_77))
        except Exception:
            pass

    # Strategy 2: Optimize 1+13 ring structure
    # Initial guess: r_outer=1.0, rot=0, jitters=0
    init_params_113 = np.array([1.0, 0.0] + [0.0]*13)
    
    if _time_left() > 10:
        try:
            res = scipy.optimize.minimize(
                _neg_score_one_center,
                init_params_113,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            best_params_113 = res.x
            best_pts_113 = _generate_one_center_params(best_params_113)
            _update_best(best_pts_113, _score(best_pts_113))
        except Exception:
            pass

    # Strategy 3: Local refinement of the best point found so far using Nelder-Mead on raw coordinates
    if best_points is not None and _time_left() > 20:
        init_flat = best_points.flatten()
        try:
            res_nm = scipy.optimize.minimize(
                lambda x: -_score(x.reshape(-1, 2)),
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            refined_pts = res_nm.x.reshape(-1, 2)
            _update_best(refined_pts, _score(refined_pts))
        except Exception:
            pass

    # Strategy 4: Random restarts from best point with small noise, refined by L-BFGS-B on raw coords
    if best_points is not None and _time_left() > 30:
        for i in range(10):
            if _time_left() < 5:
                break
            noise = rng.normal(0, 0.05, 28)
            perturbed = best_points.flatten() + noise
            try:
                res_polish = scipy.optimize.minimize(
                    lambda x: -_score(x.reshape(-1, 2)),
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                pts_polish = res_polish.x.reshape(-1, 2)
                _update_best(pts_polish, _score(pts_polish))
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
