"""Program for Heilbronn convex regions n=14 using polar coordinates with sorted angles and differential evolution."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

DEADLINE = 110.0  # Safety margin before 120s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _params_to_points(params):
    """
    Convert parameters to Cartesian coordinates.
    params: [r_0, ..., r_{N-1}, dtheta_1, ..., dtheta_{N-1}]
    theta_0 is fixed at 0.
    theta_i = sum(dtheta_1..dtheta_i)
    """
    n = N
    radii = params[:n]
    dthetas = params[n:]  # N-1 values
    
    # Build angles: theta_0 = 0, theta_i = cumsum(dthetas[:i])
    angles = np.zeros(n)
    if n > 1:
        angles[1:] = np.cumsum(dthetas)
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for minimization: negative of score."""
    points = _params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-14:
        return 1e10
    return -min_area / hull_a


def _initial_params_regular():
    """Generate initial polar parameters for a regular polygon."""
    R = 1.0
    radii = np.ones(N) * R
    dthetas = np.full(N - 1, 2.0 * np.pi / N)
    return np.concatenate([radii, dthetas])


def _local_refine(params, max_iter=100):
    """Local refinement using L-BFGS-B."""
    try:
        n = N
        r_bounds = [(0.1, 2.0)] * n
        max_dtheta = (2.0 * np.pi / N) * 1.2
        min_dtheta = 0.01
        dtheta_bounds = [(min_dtheta, max_dtheta)] * (n - 1)
        bounds = r_bounds + dtheta_bounds
        
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < _objective(params):
            return result.x
        return params
    except Exception:
        return params


def construct():
    start_time = time.time()
    
    min_dtheta = 0.05
    max_dtheta = 2.0 * np.pi / N * 1.5
    
    r_bounds = [(0.1, 2.0)] * N
    dtheta_bounds = [(min_dtheta, max_dtheta)] * (N - 1)
    bounds = r_bounds + dtheta_bounds
    
    x0 = _initial_params_regular()
    
    best_score = 0.0
    best_params = None
    
    if time.time() - start_time < DEADLINE - 30:
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=10,
                polish=False,
                init='sobol'
            )
            if result.x is not None:
                cand_pts = _params_to_points(result.x)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = result.x
        except Exception:
            pass
    
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        try:
            refined = _local_refine(best_params, max_iter=200)
            cand_pts = _params_to_points(refined)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = refined
        except Exception:
            pass
        
    if time.time() - start_time < DEADLINE - 20:
        try:
            result2 = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0,
                seed=43,
                maxiter=150,
                tol=1e-8,
                popsize=10,
                polish=False,
                init='sobol'
            )
            if result2.x is not None:
                refined2 = _local_refine(result2.x, max_iter=100)
                cand_pts = _params_to_points(refined2)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = refined2
        except Exception:
            pass
            
    if best_params is None:
        best_params = x0
        
    return _params_to_points(best_params)
# EVOLVE-BLOCK-END
