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
        if len(np.unique(points, axis=0)) < 3:
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

def _generate_7_7(r, delta):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r).
    """
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return deadline - time.time()

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 14-gon
    candidates.append(_regular_polygon(14, 1.0))
    
    # 2. 7+7 structure with a few good (r, delta) guesses
    for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for delta in [0.0, np.pi/7, 2*np.pi/7, 3*np.pi/7, np.pi/14, 3*np.pi/14]:
            candidates.append(_generate_7_7(r, delta))
            
    # 3. 13 on circle + 1 center
    center = np.array([[0.0, 0.0]])
    outer_13 = _regular_polygon(13, 1.0)
    candidates.append(np.vstack([center, outer_13]))
    
    # 4. 12 on circle + 2 centers (symmetric)
    outer_12 = _regular_polygon(12, 1.0)
    c1 = np.array([[0.5, 0.0]])
    c2 = np.array([[-0.5, 0.0]])
    candidates.append(np.vstack([outer_12, c1, c2]))
    
    # 5. Random points in unit disk
    rng = np.random.RandomState(42)
    for _ in range(20):
        pts = rng.uniform(-1, 1, (14, 2))
        # Ensure they are not all collinear
        if np.linalg.matrix_rank(pts - pts[0]) < 2:
            continue
        candidates.append(pts)

    # Phase 2: Local optimization using Nelder-Mead on EXACT score
    # This is a structural change: using derivative-free optimization on the non-smooth exact objective
    # instead of gradient-based optimization on a smooth surrogate.
    
    for i, init_pts in enumerate(candidates):
        if _time_left() < 5:
            break
            
        # Normalize initial points to unit area hull to standardize scale
        init_pts_norm = _normalize(init_pts)
        
        # Check score of initial point
        sc_init = _score(init_pts_norm)
        _update_best(init_pts_norm, sc_init)
        
        # Run Nelder-Mead
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                init_pts_norm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            new_pts = _normalize(res.x.reshape(-1, 2))
            sc_new = _score(new_pts)
            _update_best(new_pts, sc_new)
        except Exception:
            pass

    # Phase 3: Polishing with more iterations on the best found
    if best_points is not None and _time_left() > 10:
        current_pts = best_points.copy()
        
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            new_pts = _normalize(res.x.reshape(-1, 2))
            sc_new = _score(new_pts)
            _update_best(new_pts, sc_new)
        except Exception:
            pass

    # Phase 4: Random restarts from best solution if time remains
    if _time_left() > 15 and best_points is not None:
        rng = np.random.RandomState(123)
        base_pts = best_points
        
        for _ in range(20):
            if _time_left() < 5:
                break
            # Perturb the best solution
            noise = rng.normal(0, 0.2, base_pts.shape)
            init_pts = base_pts + noise
            init_pts = _normalize(init_pts)
            
            try:
                res = scipy.optimize.minimize(
                    _neg_score,
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                new_pts = _normalize(res.x.reshape(-1, 2))
                sc_new = _score(new_pts)
                _update_best(new_pts, sc_new)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
