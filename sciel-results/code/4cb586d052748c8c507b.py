"""
Optimization approach: Fourier-based smooth convex curve representation.
Parameters: low-frequency Fourier coefficients defining a smooth convex boundary,
with 14 points placed at optimized arc-length positions along this curve.
This reduces search dimensionality from 28 to ~14-16 variables.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
# Number of Fourier harmonics to use (keep low for smoothness and convexity)
N_HARMONICS = 3
# Total parameters: 2*N_HARMONICS for shape + N for point positions along curve
N_SHAPE_PARAMS = 2 * N_HARMONICS  # 6
N_TOTAL_PARAMS = N_SHAPE_PARAMS + N  # 20

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _fourier_curve(t, shape_params):
    """
    Evaluate the Fourier-based convex curve at parameter values t in [0, 2*pi).
    shape_params: array of length 2*N_HARMONICS
    Returns: (x, y) arrays of same shape as t
    """
    r = np.ones_like(t)
    for k in range(1, N_HARMONICS + 1):
        a_k = shape_params[2*(k-1)]
        b_k = shape_params[2*(k-1)+1]
        r += a_k * np.cos(k * t) + b_k * np.sin(k * t)
    # Ensure radius stays positive (convexity constraint)
    # Add a base radius to prevent negative values
    r = 1.0 + 0.3 * (r - 1.0)  # Scale down deviations to maintain convexity
    r = np.maximum(r, 0.3)  # Minimum radius
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

def _decode_params(params):
    """
    Decode optimization parameters into point coordinates.
    params: array of length N_TOTAL_PARAMS
    First N_SHAPE_PARAMS: Fourier coefficients
    Last N: normalized positions along the curve [0, 1)
    Returns: (N, 2) array of point coordinates
    """
    shape_params = params[:N_SHAPE_PARAMS]
    positions = params[N_SHAPE_PARAMS:]
    
    # Map positions [0,1) to [0, 2*pi)
    t_values = positions * 2 * np.pi
    
    # Evaluate curve at these positions
    x, y = _fourier_curve(t_values, shape_params)
    
    return np.column_stack([x, y])

def _objective_fourier(params):
    """
    Objective function for Fourier-parameterized optimization.
    Minimizes hull_area / min_triangle_area.
    """
    if not np.all(np.isfinite(params)):
        return 1e10
    
    pts = _decode_params(params)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
        
    return hull_area / min_area

def _generate_initial_fourier_configs():
    """
    Generate diverse initial configurations in Fourier parameter space.
    """
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Circle (all Fourier coefficients zero)
    circle_params = np.zeros(N_TOTAL_PARAMS)
    circle_params[N_SHAPE_PARAMS:] = np.linspace(0, 1, N, endpoint=False)
    configs.append(circle_params)
    
    # 2. Slight elliptical deformation
    for a1 in [-0.2, -0.1, 0.1, 0.2]:
        params = np.zeros(N_TOTAL_PARAMS)
        params[0] = a1  # cos(t) coefficient
        params[N_SHAPE_PARAMS:] = np.linspace(0, 1, N, endpoint=False)
        configs.append(params)
    
    # 3. Slight triangular deformation
    for a3 in [-0.1, -0.05, 0.05, 0.1]:
        params = np.zeros(N_TOTAL_PARAMS)
        params[4] = a3  # cos(3t) coefficient
        params[N_SHAPE_PARAMS:] = np.linspace(0, 1, N, endpoint=False)
        configs.append(params)
    
    # 4. Random small perturbations of circle
    for _ in range(15):
        params = np.zeros(N_TOTAL_PARAMS)
        params[:N_SHAPE_PARAMS] = rng.normal(0, 0.05, N_SHAPE_PARAMS)
        # Random positions with some jitter
        positions = np.linspace(0, 1, N, endpoint=False)
        positions += rng.normal(0, 0.02, N)
        positions = np.sort(positions) % 1.0  # Keep in [0,1)
        params[N_SHAPE_PARAMS:] = positions
        configs.append(params)
    
    # 5. Optimized position variations on circle
    for _ in range(10):
        params = np.zeros(N_TOTAL_PARAMS)
        # Slightly non-uniform positions
        base = np.linspace(0, 1, N, endpoint=False)
        jitter = rng.normal(0, 0.05, N)
        positions = np.sort((base + jitter) % 1.0)
        params[N_SHAPE_PARAMS:] = positions
        configs.append(params)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_fourier(_params_from_points(pts))
    
    def _params_from_points(pts):
        """
        Roughly estimate Fourier parameters from a point set.
        This is a heuristic for initialization only.
        """
        params = np.zeros(N_TOTAL_PARAMS)
        # Use uniform positions as default
        params[N_SHAPE_PARAMS:] = np.linspace(0, 1, N, endpoint=False)
        # Estimate shape from radial distances
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        # Fit simple Fourier to r(theta)
        for k in range(1, N_HARMONICS + 1):
            a_k = 2.0 / N * np.sum((r - np.mean(r)) * np.cos(k * theta))
            b_k = 2.0 / N * np.sum((r - np.mean(r)) * np.sin(k * theta))
            # Scale down to maintain convexity
            scale = 0.3
            params[2*(k-1)] = a_k * scale
            params[2*(k-1)+1] = b_k * scale
        return params

    # Phase 1: Multi-start Nelder-Mead in Fourier space
    inits = _generate_initial_fourier_configs()
    for i, params0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_fourier, params0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success or res.fun < best_obj:
                pts = _decode_params(res.x)
                update_best(pts)
        except Exception:
            pass

    # Phase 2: Differential Evolution in Fourier space (reduced dimensionality)
    if time.time() < deadline - 20:
        try:
            # Bounds: shape params in [-0.3, 0.3] for convexity, positions in [0, 1]
            bounds = [(-0.3, 0.3)] * N_SHAPE_PARAMS + [(0, 1)] * N
            de_res = differential_evolution(
                _objective_fourier, bounds,
                seed=42, maxiter=80, tol=1e-8,
                popsize=12, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                pts = _decode_params(de_res.x)
                update_best(pts)
        except Exception:
            pass

    # Phase 3: Final Polish with multiple methods
    if best_pts is not None and time.time() < deadline - 5:
        try:
            # Get parameters from best points
            params_best = _params_from_points(best_pts)
            
            # 1. Nelder-Mead refinement
            res = minimize(_objective_fourier, params_best, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                pts = _decode_params(res.x)
                update_best(pts)
            
            # 2. Powell's Method
            if time.time() < deadline - 2:
                res_powell = minimize(_objective_fourier, params_best, method='Powell',
                                      options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14})
                if res_powell.fun < best_obj:
                    pts = _decode_params(res_powell.x)
                    update_best(pts)
        except Exception:
            pass

    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass

    return best_pts
