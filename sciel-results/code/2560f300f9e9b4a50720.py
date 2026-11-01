import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
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

def _triangle_areas(points):
    """Compute all triangle areas (positive) for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    return _triangle_areas(points).min()

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

def _soft_min(x, beta=100.0):
    """Soft-min approximation using log-sum-exp."""
    x_min = x.min()
    shifted = x - x_min
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 50.0 / beta)
    return x_min - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _neg_smooth_score(points_flat, beta=100.0):
    """Negative smooth score for minimization - smooth surrogate for the nonsmooth min."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e6
        areas = _triangle_areas(points)
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area / ha
    except Exception:
        return 1e6

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

def _coordinate_descent_polish(points, max_iter=5000, tol=1e-10, deadline=None):
    """
    Coordinate descent focusing on the bottleneck triangle.
    At each step, identify the triangle with minimum area, then try to
    move each of its 3 vertices in small steps to increase that area
    while not decreasing the global minimum too much.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        areas = _triangle_areas(best_pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        i, j, k = _TRI_INDICES[min_idx]
        bottleneck_vertices = [i, j, k]
        
        improved = False
        for vi in bottleneck_vertices:
            # Try small perturbations of vertex vi
            base = best_pts[vi].copy()
            best_local = base.copy()
            best_local_score = best_score
            
            # Try different step sizes and directions
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        trial = best_pts.copy()
                        trial[vi] = base + np.array([dx, dy])
                        trial_score = _score(trial)
                        if trial_score > best_local_score + 1e-12:
                            best_local_score = trial_score
                            best_local = trial[vi].copy()
            
            if best_local_score > best_score + 1e-12:
                best_pts[vi] = best_local
                best_score = best_local_score
                improved = True
        
        if not improved:
            # Try slightly larger steps if no improvement
            for vi in bottleneck_vertices:
                base = best_pts[vi].copy()
                best_local = base.copy()
                best_local_score = best_score
                
                for step in [0.1, 0.2, 0.3]:
                    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                    for a in angles:
                        dx = step * np.cos(a)
                        dy = step * np.sin(a)
                        trial = best_pts.copy()
                        trial[vi] = base + np.array([dx, dy])
                        trial_score = _score(trial)
                        if trial_score > best_local_score + 1e-12:
                            best_local_score = trial_score
                            best_local = trial[vi].copy()
                
                if best_local_score > best_score + 1e-12:
                    best_pts[vi] = best_local
                    best_score = best_local_score
                    improved = True
            
            if not improved:
                break
    
    return best_pts, best_score

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, seed=42):
    """Points on two concentric circles."""
    rng = np.random.RandomState(seed)
    # 7 points on outer, 7 on inner
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return pts

def _random_in_disk(n, seed=None):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, base_radius, rotation)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def _points_from_9d_params(params):
    """
    Map 9 parameters to 14 points.
    Params:
    0: R_out
    1: R_in
    2: theta_out (rotation)
    3: theta_in (rotation)
    4-8: 5 angular perturbations for outer ring (indices 0-4)
    Inner ring is regular with rotation theta_in.
    """
    R_out = params[0]
    R_in = params[1]
    theta_out = params[2]
    theta_in = params[3]
    eps = params[4:9]
    
    # Outer ring: 7 points
    base_angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_out = base_angles_out + theta_out
    # Apply perturbations to first 5 points
    for i in range(5):
        angles_out[i] += eps[i]
    
    x_out = R_out * np.cos(angles_out)
    y_out = R_out * np.sin(angles_out)
    
    # Inner ring: 7 points (regular)
    base_angles_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_in = base_angles_in + theta_in
    
    x_in = R_in * np.cos(angles_in)
    y_in = R_in * np.sin(angles_in)
    
    pts = np.vstack([np.column_stack([x_out, y_out]), np.column_stack([x_in, y_in])])
    return pts

def _neg_score_9d(params):
    """Negative score for 9D parameter space."""
    try:
        pts = _points_from_9d_params(params)
        return -_score(pts)
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Differential Evolution on 9D space
    bounds = [
        (0.5, 2.0),  # R_out
        (0.1, 1.5),  # R_in
        (0.0, 2 * np.pi),  # theta_out
        (0.0, 2 * np.pi),  # theta_in
        (-0.5, 0.5),  # eps 0
        (-0.5, 0.5),  # eps 1
        (-0.5, 0.5),  # eps 2
        (-0.5, 0.5),  # eps 3
        (-0.5, 0.5),  # eps 4
    ]
    
    try:
        res = scipy.optimize.differential_evolution(
            _neg_score_9d, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8,
            polish=False
        )
        
        best_params = res.x
        pts = _points_from_9d_params(best_params)
        pts = _normalize(pts)
        sc = _score(pts)
        
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
            
    except Exception:
        pass
    
    # 2. Fallback: Nelder-Mead on 2D from regular polygon
    if time.time() < deadline - 10:
        init = _regular_polygon(N, 1.0, 0.0)
        init_norm = _normalize(init)
        
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                init_norm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
        except Exception:
            pass

    # 3. Polish the best solution
    if best_points is not None and time.time() < deadline - 5:
        polished, pol_score = _coordinate_descent_polish(best_points, max_iter=2000, deadline=deadline - 2)
        polished = _normalize(polished)
        final_score = _score(polished)
        if final_score > best_score:
            best_score = final_score
            best_points = polished.copy()

    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points
