import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

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

def _neg_score_full(x):
    """Negative score for full 28-dimensional point optimization."""
    try:
        pts = x.reshape(-1, 2)
        return -_score(pts)
    except Exception:
        return 1e10

def _soft_min_triangle_area(points, beta=1000.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Higher beta -> closer to true min.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = areas.max()
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - (1.0 / beta) * log_sum
    return soft_min

def _soft_score(points, beta=1000.0):
    """
    Smooth score: soft_min_triangle_area / hull_area.
    """
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return smta / ha
    except Exception:
        return 0.0

def _neg_soft_score(x, beta=1000.0):
    """Negative soft score for gradient-based optimization."""
    try:
        pts = x.reshape(-1, 2)
        return -_soft_score(pts, beta)
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

    # 1. Generate diverse initial configurations
    initial_configs = []
    
    # Regular 14-gon
    initial_configs.append(_regular_polygon(N, 1.0))
    
    # Two regular heptagons with various parameters
    for rho in [0.3, 0.5, 0.7]:
        for phi in [0, np.pi/7, 2*np.pi/7]:
            initial_configs.append(_generate_two_regular_heptagons(rho, phi))
    
    # Three rings: 5 + 4 + 5 or 6 + 2 + 6
    def _three_ring(r1, r2, r3, n1, n2, n3, phi1=0, phi2=0):
        pts = []
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + phi1
        pts.append(r1 * np.column_stack([np.cos(angles1), np.sin(angles1)]))
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
        pts.append(r2 * np.column_stack([np.cos(angles2), np.sin(angles2)]))
        angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False)
        pts.append(r3 * np.column_stack([np.cos(angles3), np.sin(angles3)]))
        return np.vstack(pts)
    
    initial_configs.append(_three_ring(0.4, 0.7, 1.0, 5, 4, 5))
    initial_configs.append(_three_ring(0.5, 0.8, 1.0, 6, 2, 6))
    
    # Random configurations
    np.random.seed(42)
    for _ in range(5):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.random.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # Evaluate all initial configs
    for pts in initial_configs:
        if _time_left() < 5.0:
            break
        sc = _score(pts)
        _update_best(pts, sc)
    
    # 2. Grid search over two-heptagon parameters
    if _time_left() > 10.0:
        n_rho = 50
        n_phi = 50
        rho_vals = np.linspace(0.2, 0.8, n_rho)
        phi_vals = np.linspace(0, 2 * np.pi / 7, n_phi, endpoint=False)
        
        best_grid_score = 0.0
        best_grid_params = (0.5, 0.0)
        
        for rho in rho_vals:
            for phi in phi_vals:
                if _time_left() < 10.0:
                    break
                sc = _score(_generate_two_regular_heptagons(rho, phi))
                if sc > best_grid_score:
                    best_grid_score = sc
                    best_grid_params = (rho, phi)
            if _time_left() < 10.0:
                break
        
        # Update best from grid
        if best_grid_score > 0:
            pts = _generate_two_regular_heptagons(*best_grid_params)
            _update_best(pts, _score(pts))
    
    # 3. Gradient-based optimization with annealed soft-min
    if _time_left() > 30.0 and best_points is not None:
        x0 = best_points.flatten()
        
        # Anneal beta from low to high for better convergence
        betas = [100.0, 500.0, 2000.0]
        current_x = x0.copy()
        
        for beta in betas:
            if _time_left() < 15.0:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, beta),
                    current_x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 1e9:  # valid result
                    current_x = res.x
                    pts = current_x.reshape(-1, 2)
                    sc = _score(pts)
                    _update_best(pts, sc)
            except Exception:
                pass
    
    # 4. Nelder-Mead from best point
    if _time_left() > 15.0 and best_points is not None:
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                _neg_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res.fun < 0:
                pts = res.x.reshape(-1, 2)
                sc = _score(pts)
                _update_best(pts, sc)
        except Exception:
            pass
    
    # 5. Multi-start Nelder-Mead from diverse configs
    if _time_left() > 20.0:
        # Use top 3 distinct initial configs
        scored_configs = []
        for pts in initial_configs[:10]:
            sc = _score(pts)
            scored_configs.append((sc, pts))
        scored_configs.sort(key=lambda x: -x[0])
        
        for sc, pts in scored_configs[:3]:
            if _time_left() < 10.0:
                break
            try:
                x0 = pts.flatten()
                res = scipy.optimize.minimize(
                    _neg_score_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res.fun < 0:
                    new_pts = res.x.reshape(-1, 2)
                    new_sc = _score(new_pts)
                    _update_best(new_pts, new_sc)
            except Exception:
                pass
    
    # 6. Final polish
    if _time_left() > 5.0 and best_points is not None:
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                _neg_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 0:
                pts = res.x.reshape(-1, 2)
                sc = _score(pts)
                _update_best(pts, sc)
        except Exception:
            pass
    
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    else:
        best_points = _normalize(best_points)
        
    return best_points
