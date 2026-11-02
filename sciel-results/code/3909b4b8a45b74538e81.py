import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
MAX_TIME = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / hull area."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    ha = _hull_area_fast(points)
    if ha < 1e-9:
        return 0.0
        
    ma = _min_triangle_area_vec(points)
    if ma < 1e-9:
        return 0.0
        
    return ma / ha

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _fourier_to_points(coeffs, n_points=N):
    """
    Convert Fourier coefficients to points.
    coeffs: [a1, b1, a2, b2, ..., aK, bK]
    r(theta) = 1 + sum (a_k cos(k theta) + b_k sin(k theta))
    """
    K = len(coeffs) // 2
    if K == 0:
        theta = 2 * np.pi * np.arange(n_points) / n_points
        return np.column_stack([np.cos(theta), np.sin(theta)])
        
    theta = 2 * np.pi * np.arange(n_points) / n_points
    
    # Compute radius for each point
    r = np.ones(n_points)
    for k in range(1, K + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
        
    # Ensure convexity/validity: r must be positive.
    # A stricter condition for convexity of the curve is r + r'' > 0,
    # but for small perturbations, r > 0 is a good heuristic to avoid self-intersection.
    if np.any(r <= 0.1):
        return None
        
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _objective_soft_fourier(coeffs, beta=100.0):
    """
    Objective for Fourier coefficient optimization.
    Minimizes hull_area / soft_min(triangle_areas).
    """
    pts = _fourier_to_points(coeffs)
    if pts is None:
        return 1e10
        
    ha = _hull_area_fast(pts)
    if ha < 1e-9:
        return 1e10
        
    areas = _all_triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    # Soft min calculation
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    return ha / soft_min_val

def _objective_hard_fourier(coeffs):
    """
    Objective for Fourier coefficient optimization (hard min).
    Minimizes hull_area / min(triangle_areas).
    """
    pts = _fourier_to_points(coeffs)
    if pts is None:
        return 1e10
        
    ha = _hull_area_fast(pts)
    if ha < 1e-9:
        return 1e10
        
    ma = _min_triangle_area_vec(pts)
    if ma < 1e-12:
        return 1e10
        
    return ha / ma

def _optimize_fourier(start_coeffs, deadline, method='L-BFGS-B', obj_func=_objective_soft_fourier):
    """Run optimization on Fourier coefficients."""
    if time.time() > deadline:
        return None
        
    try:
        if method == 'L-BFGS-B':
            result = minimize(
                obj_func, 
                start_coeffs, 
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
        elif method == 'Nelder-Mead':
            result = minimize(
                obj_func, 
                start_coeffs, 
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        else:
            return None
            
        return result.x
    except Exception:
        return None

def _coordinate_descent(pts, deadline, max_rounds=20):
    """
    Cyclic coordinate descent on the actual (x,y) coordinates.
    This refines the points directly, potentially breaking the Fourier symmetry
    if it leads to a better score, but we must be careful not to destroy convexity.
    Since the score function penalizes non-convex hulls implicitly (by using hull area),
    and we start from a convex configuration, small steps should be safe.
    """
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    best_score = _score(pts)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(20):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    
                    if s_plus > best_score:
                        pts[i, coord] = orig + step
                        best_score = s_plus
                        improved = True
                    elif s_minus > best_score:
                        pts[i, coord] = orig - step
                        best_score = s_minus
                        improved = True
                    else:
                        break
                    
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
            
    return pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME
    
    # Baseline: Regular 14-gon
    best_pts = _fourier_to_points(np.zeros(0))
    best_score = _score(best_pts)
    
    # Strategy:
    # 1. Optimize Fourier coefficients using L-BFGS-B with soft objective.
    # 2. Multi-start with small random perturbations.
    # 3. Refine the best candidate using coordinate descent on coordinates.
    
    K = 4 # Use 4 harmonics (8 coefficients) for more flexibility
    n_coeffs = 2 * K
    
    # Start 1: Zero coefficients (Regular polygon)
    start_coeffs = np.zeros(n_coeffs)
    res_coeffs = _optimize_fourier(start_coeffs, deadline, method='L-BFGS-B', obj_func=_objective_soft_fourier)
    
    if res_coeffs is not None:
        pts = _fourier_to_points(res_coeffs)
        if pts is not None:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    # Multi-start with random perturbations
    rng = np.random.default_rng(42)
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Random perturbations
        # Scale 0.05 is small enough to keep convexity mostly intact
        start_coeffs = rng.uniform(-0.05, 0.05, n_coeffs)
        
        # Use Nelder-Mead for some starts to escape local minima of the soft objective?
        # Or stick with L-BFGS for speed. Let's alternate.
        method = 'L-BFGS-B' if i % 2 == 0 else 'Nelder-Mead'
        obj_func = _objective_soft_fourier if method == 'L-BFGS-B' else _objective_hard_fourier
        
        res_coeffs = _optimize_fourier(start_coeffs, deadline, method=method, obj_func=obj_func)
        
        if res_coeffs is not None:
            pts = _fourier_to_points(res_coeffs)
            if pts is not None:
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts

    # Final Refinement: Coordinate Descent
    # We have a good candidate in best_pts. Let's refine it directly in coordinate space.
    if time.time() < deadline - 10:
        refined_pts, refined_score = _coordinate_descent(best_pts, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts

    # Ensure we return a valid configuration
    if best_pts is None or _score(best_pts) < 1e-6:
        best_pts = _fourier_to_points(np.zeros(0))
        
    # Normalize to unit area
    ha = _hull_area_fast(best_pts)
    if ha > 1e-9:
        best_pts = best_pts * np.sqrt(1.0 / ha)
        
    return best_pts

# EVOLVE-BLOCK-END
