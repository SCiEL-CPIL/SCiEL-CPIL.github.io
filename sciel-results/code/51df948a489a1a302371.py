import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    if n_out + n_in != N:
        return None
        
    pts_out = _make_regular_polygon(n_out, 1.0, 0.0)
    pts_in = _make_regular_polygon(n_in, r, theta)
    
    return np.vstack([pts_out, pts_in])

def _neg_ring_score(params, n_out, n_in):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.01 or r > 0.99:
        return 0.0
        
    points = _generate_ring_points(n_out, n_in, r, theta)
    if points is None:
        return 0.0
        
    s = _score(points)
    return -s

def _optimize_ring_split(n_out, n_in, time_budget):
    """
    Optimize a specific split (n_out, n_in) using DE + Local Polish.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.05, 0.95), (0.0, theta_max)]
    
    best_score = -np.inf
    best_points = None
    
    popsize = 15
    maxiter = 50
    
    try:
        res = differential_evolution(
            lambda p: _neg_ring_score(p, n_out, n_in),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=True,
        )
        
        if res.fun < 0: # Valid score
            best_score = -res.fun
            best_params = res.x
            best_points = _generate_ring_points(n_out, n_in, best_params[0], best_params[1])
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None

    if best_points is None or best_score < 0.5:
        heuristic_starts = []
        for r in [0.3, 0.5, 0.7, 0.9]:
            for theta in [0, theta_max/2, theta_max/3]:
                heuristic_starts.append([r, theta])
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_ring_score(start, n_out, n_in)
            if s > best_score:
                best_score = s
                best_points = _generate_ring_points(n_out, n_in, start[0], start[1])
                best_params = start

    if best_points is not None and time.time() < deadline - 5:
        try:
            res_opt = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            opt_points = res_opt.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
                
        except Exception:
            pass
            
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_opt2 = minimize(
                _neg_score,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 1000,
                    'maxfev': 3000
                }
            )
            opt_points2 = res_opt2.x.reshape((N, 2))
            opt_score2 = _score(opt_points2)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2
        except Exception:
            pass

    return best_score, best_points

# New Superellipse Logic

def _superellipse_points(n_shape, aspect, rotation, n_pts=N):
    """
    Generate n_pts points on a superellipse |x/a|^n + |y/b|^n = 1.
    n_shape: The exponent n (Lamé exponent). n=2 is circle, n->inf is square.
    aspect: Ratio a/b.
    rotation: Rotation angle.
    """
    if n_shape <= 0:
        n_shape = 1e-6
    
    # Sample angles uniformly for arc length approximation?
    # Actually, uniform angle sampling on superellipse is not uniform arc length.
    # But for optimization, we can just optimize the parameters and then polish coordinates.
    # Let's sample angles uniformly first.
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    
    # Superellipse parametric equations:
    # x = a * sign(cos(t)) * |cos(t)|^(2/n)
    # y = b * sign(sin(t)) * |sin(t)|^(2/n)
    
    # To avoid issues with 0^0 or sign(0), we use a robust calculation.
    # For n=2, this reduces to standard circle.
    
    # Let's normalize such that max radius is 1.
    # If we use a=1, b=1/aspect, then max x is 1, max y is 1/aspect.
    # We want to keep the scale consistent for scoring (score is normalized by hull area, so scale doesn't matter for score, but helps optimization stability).
    # Let's fix a=1.
    
    a = 1.0
    b = 1.0 / aspect if aspect > 0 else 1.0
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Handle sign and power
    # x = a * sign(cos) * |cos|^(2/n)
    # y = b * sign(sin) * |sin|^(2/n)
    
    # If n is very large, 2/n is small, |cos|^(small) is close to 1 for non-zero cos.
    # This creates a box-like shape.
    
    x = a * np.sign(cos_t) * np.abs(cos_t)**(2.0/n_shape)
    y = b * np.sign(sin_t) * np.abs(sin_t)**(2.0/n_shape)
    
    # Handle points exactly at 0 to avoid 0^small issues (though 0^positive is 0)
    # np.sign(0) is 0, so x becomes 0. This is correct.
    
    pts = np.column_stack([x, y])
    
    # Rotate
    c = np.cos(rotation)
    s = np.sin(rotation)
    rot_matrix = np.array([[c, -s], [s, c]])
    pts = pts @ rot_matrix.T
    
    return pts

def _neg_superellipse_score(params):
    """
    Objective for superellipse parameters.
    params: [n_shape, aspect, rotation]
    """
    n_shape, aspect, rotation = params
    
    # Bounds check
    if n_shape < 1.0 or n_shape > 20.0:
        return 0.0
    if aspect < 0.5 or aspect > 2.0:
        return 0.0
    if rotation < 0 or rotation > np.pi/4:
        return 0.0
        
    pts = _superellipse_points(n_shape, aspect, rotation)
    s = _score(pts)
    return -s

def _optimize_superellipse(time_budget):
    """
    Optimize superellipse parameters.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    bounds = [
        (1.0, 20.0),  # n_shape
        (0.5, 2.0),   # aspect
        (0.0, np.pi/4) # rotation (symmetry allows 0 to 90 deg, but 0-45 is enough due to reflection)
    ]
    
    best_score = -np.inf
    best_points = None
    
    # Differential Evolution
    try:
        res = differential_evolution(
            _neg_superellipse_score,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=True
        )
        
        if res.fun < 0:
            best_score = -res.fun
            best_params = res.x
            best_points = _superellipse_points(best_params[0], best_params[1], best_params[2])
    except Exception:
        pass
    
    # Heuristic starts if DE failed or score low
    if best_points is None or best_score < 0.5:
        heuristic_starts = [
            [2.0, 1.0, 0.0], # Circle
            [4.0, 1.0, 0.0], # Rounded square
            [10.0, 1.0, 0.0], # Square-ish
            [2.0, 1.2, 0.0], # Ellipse
            [4.0, 0.8, 0.0]
        ]
        
        for start in heuristic_starts:
            if time.time() > deadline - 5:
                break
            s = -_neg_superellipse_score(start)
            if s > best_score:
                best_score = s
                best_points = _superellipse_points(start[0], start[1], start[2])
    
    # Coordinate Polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_opt = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            opt_points = res_opt.x.reshape((N, 2))
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            pass
            
    return best_score, best_points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    np.random.seed(42)
    
    best_global_score = -np.inf
    best_global_points = None
    
    # Baseline: Regular polygon
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_global_score = s_reg
    best_global_points = reg_poly.copy()
    
    # 1. Try Superellipse Optimization
    if time.time() < DEADLINE - 60:
        s_se, pts_se = _optimize_superellipse(60.0)
        if pts_se is not None and s_se > best_global_score:
            best_global_score = s_se
            best_global_points = pts_se
            
    # 2. Try Ring Optimization (Fallback/Comparison)
    # We can try the best ring split (7,7) if time permits
    if time.time() < DEADLINE - 40:
        s_ring, pts_ring = _optimize_ring_split(7, 7, 40.0)
        if pts_ring is not None and s_ring > best_global_score:
            best_global_score = s_ring
            best_global_points = pts_ring
            
    # 3. Final Polish on the absolute best
    if best_global_points is not None and time.time() < DEADLINE - 10:
        try:
            # One last high-precision polish
            res_final = minimize(
                _neg_score,
                best_global_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            final_points = res_final.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_points = final_points
        except Exception:
            pass

    return best_global_points
