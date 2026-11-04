import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0

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

def _fourier_points(coeffs):
    """
    Generate points from Fourier coefficients.
    coeffs: array of shape (2, K) where K is number of harmonics.
    coeffs[0, k] = cos coefficient for harmonic k
    coeffs[1, k] = sin coefficient for harmonic k
    Point i: (1 + sum_k (a_k cos(k theta_i) + b_k sin(k theta_i)), 
                  sum_k (a_k cos(k theta_i) + b_k sin(k theta_i)))
    Wait, better parameterization:
    x_i = sum_{k=0}^{K-1} (A_k cos(k theta_i) + B_k sin(k theta_i))
    y_i = sum_{k=0}^{K-1} (C_k cos(k theta_i) + D_k sin(k theta_i))
    
    To ensure convexity and reasonable shape, we can constrain the radial function:
    r(theta) = 1 + sum_{k=1}^{K-1} (a_k cos(k theta) + b_k sin(k theta))
    x_i = r(theta_i) cos(theta_i)
    y_i = r(theta_i) sin(theta_i)
    
    This ensures points are on a star-shaped curve around origin.
    If coefficients are small, it's close to a circle.
    """
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # coeffs shape: (K, 2) where K is number of harmonics (excluding DC)
    # coeffs[k] = [a_k, b_k]
    K = coeffs.shape[0]
    
    r = np.ones(N)
    for k in range(1, K+1):
        a_k = coeffs[k-1, 0]
        b_k = coeffs[k-1, 1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
        
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack([x, y])

def _neg_fourier_score(coeffs_flat):
    """
    Objective for Fourier coefficients.
    coeffs_flat: flattened array of shape (2*K,)
    """
    K = coeffs_flat.size // 2
    coeffs = coeffs_flat.reshape((K, 2))
    points = _fourier_points(coeffs)
    s = _score(points)
    return -s

def _optimize_fourier(time_budget):
    """
    Optimize Fourier coefficients using DE + Polish.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Try different numbers of harmonics
    best_score = -np.inf
    best_points = None
    
    for K in [3, 5, 7, 10]:
        if time.time() > deadline - 10:
            break
            
        bounds = [(-0.2, 0.2)] * (2*K)
        
        try:
            res = differential_evolution(
                _neg_fourier_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            
            if res.fun < 0:
                s = -res.fun
                coeffs = res.x.reshape((K, 2))
                points = _fourier_points(coeffs)
                
                if s > best_score:
                    best_score = s
                    best_points = points
                    
        except Exception:
            pass
            
    # Polish in full coordinates
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
            opt_points2 = res_opt2.reshape((N, 2))
            opt_score2 = _score(opt_points2)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2
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
    
    # 1. Try Fourier Optimization (Primary Strategy)
    if time.time() < DEADLINE - 60:
        s_fourier, pts_fourier = _optimize_fourier(60.0)
        if pts_fourier is not None and s_fourier > best_global_score:
            best_global_score = s_fourier
            best_global_points = pts_fourier
            
    # 2. Try Ring Optimization (Fallback/Comparison)
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
