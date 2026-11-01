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

def _all_triangle_areas(points):
    """Compute all triangle areas for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Numerically stable version.
    """
    areas = _all_triangle_areas(points)
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
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

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft-score for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _neg_soft_score_with_penalty(points_flat, beta=100.0, penalty_weight=1.0):
    """
    Negative soft-score with penalty for points outside a reasonable bound.
    This helps keep the optimization well-conditioned.
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e10
        
        # Center the points
        center = np.mean(points, axis=0)
        centered = points - center
        max_radius = np.max(np.sqrt(np.sum(centered**2, axis=1)))
        
        # Penalty if points spread too much
        if max_radius > 2.0:
            penalty = penalty_weight * (max_radius - 2.0)**2
        else:
            penalty = 0.0
        
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha) + penalty
    except Exception:
        return 1e10

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

def _generate_two_ring_points(r_inner, delta_inner):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r_inner).
    """
    n_ring = 7
    angles_outer = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = angles_outer + delta_inner
    points_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _generate_three_ring_points(r1, r2, d1, d2):
    """
    Generate 14 points on three rings: 6+5+3 or 5+5+4, etc.
    For simplicity, use 5+5+4 split.
    """
    # 5 points on outer ring (radius 1)
    angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts_5 = np.column_stack([np.cos(angles_5), np.sin(angles_5)])
    
    # 5 points on middle ring (radius r1)
    angles_5b = np.linspace(d1, d1 + 2 * np.pi, 5, endpoint=False)
    pts_5b = np.column_stack([r1 * np.cos(angles_5b), r1 * np.sin(angles_5b)])
    
    # 4 points on inner ring (radius r2)
    angles_4 = np.linspace(d2, d2 + 2 * np.pi, 4, endpoint=False)
    pts_4 = np.column_stack([r2 * np.cos(angles_4), r2 * np.sin(angles_4)])
    
    return np.vstack([pts_5, pts_5b, pts_4])

def _generate_grid_points(n_rows, n_cols, width, height):
    """Generate points in a grid pattern."""
    x = np.linspace(-width/2, width/2, n_cols)
    y = np.linspace(-height/2, height/2, n_rows)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.flatten(), yy.flatten()])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # Generate diverse initial configurations
    rng = np.random.RandomState(42)
    inits = []
    
    # Regular 14-gon
    inits.append(_regular_polygon(14, 1.0, 0.0))
    
    # Two-ring configurations
    for r in [0.3, 0.5, 0.7, 0.9]:
        for d in [0, np.pi/7, 2*np.pi/7, np.pi/14]:
            inits.append(_generate_two_ring_points(r, d))
    
    # Three-ring configurations
    for r1 in [0.5, 0.7]:
        for r2 in [0.3, 0.5]:
            for d1 in [0, np.pi/5]:
                for d2 in [0, np.pi/4]:
                    inits.append(_generate_three_ring_points(r1, r2, d1, d2))
    
    # Grid-like configurations
    inits.append(_generate_grid_points(4, 4, 2.0, 2.0))  # 16 points, take 14
    inits.append(_generate_grid_points(2, 7, 2.0, 1.0))
    inits.append(_generate_grid_points(7, 2, 1.0, 2.0))
    
    # Random configurations
    for _ in range(10):
        pts = rng.uniform(-1, 1, (14, 2))
        # Ensure some spread
        if np.std(pts, axis=0).min() < 0.1:
            pts = rng.uniform(-1, 1, (14, 2))
        inits.append(pts)
    
    # Evaluate all initial configurations
    init_scores = []
    for pts in inits:
        sc = _score(_normalize(pts))
        init_scores.append(sc)
        _update_best(pts, sc)
    
    # Sort by score, keep top candidates
    sorted_idx = np.argsort(init_scores)[::-1]
    top_inits = [inits[i] for i in sorted_idx[:20]]
    
    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0, 0.0))
    
    # Phase 1: L-BFGS-B optimization from each top initial config
    # Use progressive beta for smoothness
    for beta in [30.0, 80.0, 150.0]:
        if _time_left() < 10:
            break
        
        for init_pts in top_inits[:10]:  # Top 10 inits
            if _time_left() < 5:
                break
            
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score_with_penalty(x, b, penalty_weight=0.1),
                    init_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
                )
                pts_opt = _normalize(res.x.reshape(-1, 2))
                sc_opt = _score(pts_opt)
                _update_best(pts_opt, sc_opt)
            except Exception:
                pass
    
    if _time_left() < 10:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0, 0.0))
    
    # Phase 2: More aggressive optimization from best points found
    if best_points is not None:
        for beta in [100.0, 200.0, 400.0]:
            if _time_left() < 5:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
                )
                pts_opt = _normalize(res.x.reshape(-1, 2))
                sc_opt = _score(pts_opt)
                _update_best(pts_opt, sc_opt)
            except Exception:
                pass
    
    if _time_left() < 8:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0, 0.0))
    
    # Phase 3: Nelder-Mead from diverse starts for global exploration
    nm_inits = [inits[i] for i in sorted_idx[:8]]
    for init_pts in nm_inits:
        if _time_left() < 5:
            break
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                init_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            pts_opt = _normalize(res.x.reshape(-1, 2))
            sc_opt = _score(pts_opt)
            _update_best(pts_opt, sc_opt)
        except Exception:
            pass
    
    if _time_left() < 5:
        return best_points if best_points is not None else _normalize(_regular_polygon(14, 1.0, 0.0))
    
    # Phase 4: Final high-precision polish
    if best_points is not None:
        for beta in [200.0, 500.0]:
            if _time_left() < 3:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1000}
                )
                pts_opt = _normalize(res.x.reshape(-1, 2))
                sc_opt = _score(pts_opt)
                _update_best(pts_opt, sc_opt)
            except Exception:
                pass
    
    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
