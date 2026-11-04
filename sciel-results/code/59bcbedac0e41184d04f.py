import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution, minimize_scalar

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

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_heptagons(r, theta):
    """
    Generate two concentric heptagons.
    Outer ring: radius 1, 7 points starting at angle 0.
    Inner ring: radius r, 7 points rotated by theta.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + theta
    
    p_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    p_inner = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([p_outer, p_inner])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    return -_score_two_heptagons(r, theta)

def _three_ring_points(r1, r2, theta1, theta2, n1=5, n2=4, n3=5):
    """
    Generate three concentric rings with different point counts.
    Ring 1: n1 points at radius r1, angle offset 0
    Ring 2: n2 points at radius r2, angle offset theta1
    Ring 3: n3 points at radius 1.0, angle offset theta2
    """
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + theta1
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + theta2
    
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    p3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
    
    return np.vstack([p1, p2, p3])

def _score_three_ring(params):
    """Score for three-ring configuration."""
    r1, r2, theta1, theta2 = params
    pts = _three_ring_points(r1, r2, theta1, theta2)
    return _score(pts)

def _neg_score_three_ring(params):
    """Negative score for three-ring optimization."""
    return -_score_three_ring(params)

def _local_optimize(points, time_limit=20.0):
    """Nelder-Mead refinement."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
        
    return best_score, best_pts

def _optimize_structure_two_hept(time_limit=20.0):
    """
    Optimize the two-heptagon structure (r, theta).
    """
    start_time = time.time()
    
    best_r = 0.5
    best_theta = np.pi / 7
    best_score = _score_two_heptagons(best_r, best_theta)
    
    def _neg_score_2d(x):
        if time.time() - start_time > time_limit:
            return 1000.0
        r, theta = x
        return -_score_two_heptagons(r, theta)
    
    x0 = np.array([best_r, best_theta])
    
    try:
        res = minimize(
            _neg_score_2d, x0, method='Nelder-Mead',
            bounds=[(0.1, 0.9), (0, 2*np.pi)],
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if time.time() - start_time < time_limit:
            r_opt, theta_opt = res.x
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_r = r_opt
                best_theta = theta_opt
    except Exception:
        pass
    
    for theta_init in [0, np.pi/7, 2*np.pi/7, np.pi/14]:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            res_r = minimize_scalar(
                lambda r: -_score_two_heptagons(r, theta_init),
                bounds=(0.1, 0.9),
                method='bounded',
                options={'xatol': 1e-6}
            )
            r_opt = res_r.x
            s_opt = -res_r.fun
            if s_opt > best_score:
                best_score = s_opt
                best_r = r_opt
                best_theta = theta_init
        except Exception:
            continue

    return best_score, best_r, best_theta

def _optimize_structure_three_ring(time_limit=20.0):
    """
    Optimize the three-ring structure (r1, r2, theta1, theta2).
    """
    start_time = time.time()
    
    # Initial guess
    best_params = np.array([0.4, 0.7, np.pi/5, np.pi/5])
    best_score = _score_three_ring(best_params)
    
    def _neg_score_4d(x):
        if time.time() - start_time > time_limit:
            return 1000.0
        return -_score_three_ring(x)
    
    try:
        res = minimize(
            _neg_score_4d, best_params, method='Nelder-Mead',
            bounds=[(0.1, 0.9), (0.1, 0.9), (0, 2*np.pi), (0, 2*np.pi)],
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if time.time() - start_time < time_limit:
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Try a few different initial conditions
    for init in [
        np.array([0.3, 0.6, 0.0, 0.0]),
        np.array([0.5, 0.7, np.pi/5, np.pi/5]),
        np.array([0.4, 0.8, np.pi/10, np.pi/10]),
    ]:
        if time.time() - start_time > time_limit - 5:
            break
        try:
            res = minimize(
                _neg_score_4d, init, method='Nelder-Mead',
                bounds=[(0.1, 0.9), (0.1, 0.9), (0, 2*np.pi), (0, 2*np.pi)],
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_params = res.x.copy()
        except Exception:
            continue
    
    return best_score, best_params

def _differential_evolution_search(points_init, time_limit=15.0):
    """
    Short differential evolution over full 28D space.
    """
    start_time = time.time()
    
    # Define bounds: normalize to unit circle roughly
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1000.0
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points_init)
    best_pts = points_init.copy()
    
    try:
        res = differential_evolution(
            obj, bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            update_interval=1,
            callback=lambda xk, convergence: None
        )
        if time.time() - start_time < time_limit:
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Try the structured Two-Heptagon optimization
    if time.time() < deadline - 40:
        s_struct, r_opt, theta_opt = _optimize_structure_two_hept(time_limit=20.0)
        if s_struct > best_overall_score:
            best_overall_score = s_struct
            best_overall_points = _two_heptagons(r_opt, theta_opt)
    
    # 2. Try the Three-Ring structure
    if time.time() < deadline - 30:
        s_3ring, params_3ring = _optimize_structure_three_ring(time_limit=20.0)
        if s_3ring > best_overall_score:
            best_overall_score = s_3ring
            best_overall_points = _three_ring_points(params_3ring[0], params_3ring[1], params_3ring[2], params_3ring[3])
    
    # 3. Local optimization on the best structured points
    if time.time() < deadline - 25 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=20.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # 4. Differential evolution for diversification
    if time.time() < deadline - 15 and best_overall_points is not None:
        s_de, p_de = _differential_evolution_search(best_overall_points, time_limit=15.0)
        if s_de > best_overall_score:
            best_overall_score = s_de
            best_overall_points = p_de
    
    # 5. Final Polish with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=max(5, remaining_time - 5))
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # Fallback: Regular polygon if structure failed
    if best_overall_points is None:
        init_pts = _regular_polygon(N, 1.0)
        s, p = _local_optimize(init_pts, time_limit=10.0)
        best_overall_score = s
        best_overall_points = p

    return best_overall_points
