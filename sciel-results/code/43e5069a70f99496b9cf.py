import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 14

# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    if r <= 0 or r > 1:
        return 1.0
    return -_score_two_heptagons(r, theta)

def _optimize_1d_rotation(r, time_limit=5.0):
    """Optimize rotation angle for a fixed inner radius."""
    start_time = time.time()
    
    def neg_theta(theta):
        return -_score_two_heptagons(r, theta)
    
    best_score = 0.0
    best_theta = 0.0
    
    # Coarse scan over the fundamental domain of the 7-fold symmetry
    n_scan = 50
    thetas = np.linspace(0, np.pi / 7, n_scan, endpoint=False)
    for th in thetas:
        s = _score_two_heptagons(r, th)
        if s > best_score:
            best_score = s
            best_theta = th
    
    # Fine local optimization around best
    if time.time() < start_time + time_limit:
        try:
            res = minimize_scalar(
                neg_theta,
                bounds=(max(0, best_theta - np.pi / 28), min(np.pi / 7, best_theta + np.pi / 28)),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if time.time() < start_time + time_limit:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_theta = res.x
        except Exception:
            pass
    
    return best_score, best_theta

def _optimize_radius_1d(theta, time_limit=5.0):
    """Optimize inner radius for a fixed rotation angle."""
    start_time = time.time()
    
    def neg_r(r):
        return -_score_two_heptagons(r, theta)
    
    best_score = 0.0
    best_r = 0.5
    
    # Coarse scan
    n_scan = 30
    radii = np.linspace(0.1, 0.95, n_scan)
    for rr in radii:
        s = _score_two_heptagons(rr, theta)
        if s > best_score:
            best_score = s
            best_r = rr
    
    # Fine local optimization
    if time.time() < start_time + time_limit:
        try:
            res = minimize_scalar(
                neg_r,
                bounds=(max(0.05, best_r - 0.1), min(0.98, best_r + 0.1)),
                method='bounded',
                options={'xatol': 1e-8}
            )
            if time.time() < start_time + time_limit:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_r = res.x
        except Exception:
            pass
    
    return best_score, best_r

def _local_optimize_free(points, time_limit=10.0):
    """Local optimization in full 2D space."""
    start_time = time.time()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if time.time() < start_time + time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Two concentric heptagons with joint (r, theta) optimization
    # Start with a grid of (r, theta) values
    r_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    theta_values = [0.0, np.pi / 14, np.pi / 7, 2 * np.pi / 14, 3 * np.pi / 14, np.pi / 4]
    
    # Coarse grid search
    for r in r_values:
        if time.time() > deadline - 30:
            break
        for theta in theta_values:
            if time.time() > deadline - 25:
                break
            s = _score_two_heptagons(r, theta)
            if s > best_score:
                best_score = s
                best_points = _two_heptagons(r, theta).copy()
    
    # Refine: for each promising (r, theta), do 1D optimization
    if best_points is not None and time.time() < deadline - 40:
        # Extract current r and theta from best_points
        pts = best_points
        norms = np.linalg.norm(pts, axis=1)
        inner_mask = norms < 0.99
        if np.sum(inner_mask) == 7:
            r_est = np.mean(norms[inner_mask])
            inner_pts = pts[inner_mask]
            angles_in = np.arctan2(inner_pts[:, 1], inner_pts[:, 0])
            outer_mask = ~inner_mask
            outer_pts = pts[outer_mask]
            angles_out = np.arctan2(outer_pts[:, 1], outer_pts[:, 0])
            theta_est = (np.mean(angles_in) - np.mean(angles_out)) % (2 * np.pi)
        else:
            r_est = 0.5
            theta_est = 0.0
        
        # Optimize theta for current r
        s_theta, theta_opt = _optimize_1d_rotation(r_est, time_limit=8.0)
        if s_theta > best_score:
            best_score = s_theta
            best_points = _two_heptagons(r_est, theta_opt).copy()
        
        # Optimize r for current theta
        s_r, r_opt = _optimize_radius_1d(theta_opt, time_limit=8.0)
        if s_r > best_score:
            best_score = s_r
            best_points = _two_heptagons(r_opt, theta_opt).copy()
        
        # Final joint refinement
        if time.time() < deadline - 20:
            s_final, theta_final = _optimize_1d_rotation(r_opt, time_limit=5.0)
            if s_final > best_score:
                best_score = s_final
                best_points = _two_heptagons(r_opt, theta_final).copy()
    
    # Strategy 2: Local optimization in full 2D space
    if best_points is not None and time.time() < deadline - 10:
        s, p = _local_optimize_free(best_points, time_limit=15.0)
        if s > best_score:
            best_score = s
            best_points = p
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
