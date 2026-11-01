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

def _generate_two_regular_heptagons(rho, phi):
    """
    Generate 14 points: 7 outer (radius 1) and 7 inner (radius rho).
    Outer angles: 2*pi*i/7
    Inner angles: 2*pi*j/7 + phi
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phi
    
    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    pts_inner = rho * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([pts_outer, pts_inner])

def _score_parameterized(params):
    """
    Score for parameterized 2-regular-heptagon configuration.
    params: [rho, phi]
    """
    try:
        rho, phi = params
        if rho < 0 or rho > 1:
            return 0.0
        points = _generate_two_regular_heptagons(rho, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_parameterized(params):
    return -_score_parameterized(params)

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

    # 1. Grid Search over rho and phi
    # rho in [0.1, 0.9], phi in [0, 2*pi/7)
    # Due to symmetry, we can restrict phi further?
    # The configuration is invariant under rotation of the whole system, so phi is relative.
    # Also, swapping inner and outer isn't possible as outer is fixed radius 1.
    # But there might be symmetry in phi.
    # Let's use a reasonably fine grid.
    
    n_rho = 100
    n_phi = 100
    rho_vals = np.linspace(0.1, 0.9, n_rho)
    phi_vals = np.linspace(0, 2 * np.pi / 7, n_phi, endpoint=False)
    
    best_grid_score = 0.0
    best_grid_params = (0.5, 0.0)
    
    for rho in rho_vals:
        for phi in phi_vals:
            if _time_left() < 5.0:
                break
            sc = _score_parameterized((rho, phi))
            if sc > best_grid_score:
                best_grid_score = sc
                best_grid_params = (rho, phi)
        
        if _time_left() < 5.0:
            break

    # 2. Local Optimization (Nelder-Mead) from best grid point
    if _time_left() > 5.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score_parameterized,
                best_grid_params,
                method='Nelder-Mead',
                bounds=[(0.01, 0.99), (0, 2 * np.pi / 7)],
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 0:
                opt_params = res.x
                # Ensure params are within bounds
                opt_params[0] = np.clip(opt_params[0], 0.01, 0.99)
                opt_params[1] = np.clip(opt_params[1], 0, 2 * np.pi / 7)
                
                points = _generate_two_regular_heptagons(opt_params[0], opt_params[1])
                sc = _score(points)
                _update_best(points, sc)
        except Exception:
            pass

    # 3. Fallback / Initial Guess: Regular 14-gon or simple 2-ring
    if best_points is None:
        # Try regular 14-gon
        init_points = _regular_polygon(N, 1.0)
        sc = _score(init_points)
        _update_best(init_points, sc)
        
        # Try simple 2-ring (7+7) with rho=0.5, phi=pi/7
        points = _generate_two_regular_heptagons(0.5, np.pi/7)
        sc = _score(points)
        _update_best(points, sc)

    # 4. Full-space local optimization from best found point
    # This allows the points to deviate from perfect regularity if it helps.
    if _time_left() > 10.0 and best_points is not None:
        try:
            # Reshape to flat array for Nelder-Mead
            x0 = best_points.flatten()
            
            def _neg_score_full(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            res_full = scipy.optimize.minimize(
                _neg_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            
            if res_full.fun < 0:
                polished_points = res_full.x.reshape(-1, 2)
                polished_score = _score(polished_points)
                _update_best(polished_points, polished_score)
        except Exception:
            pass

    # 5. Final Polish
    if _time_left() > 3.0 and best_points is not None:
        try:
            x0 = best_points.flatten()
            def _neg_score_full(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            res_final = scipy.optimize.minimize(
                _neg_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if res_final.fun < 0:
                final_points = res_final.x.reshape(-1, 2)
                final_score = _score(final_points)
                _update_best(final_points, final_score)
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    else:
        best_points = _normalize(best_points)
        
    return best_points
