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


def _polar_to_cartesian(params):
    """
    Convert parameters to Cartesian coordinates.
    params: [r_0, r_1, ..., r_{N-1}, dtheta_1, dtheta_2, ..., dtheta_{N-1}]
    where theta_0 = 0 and theta_i = sum(dtheta_1..dtheta_i)
    Total parameters: N + (N-1) = 2N-1
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
    points = _polar_to_cartesian(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-14:
        return 1e10
    return -min_area / hull_a


def _objective_soft(params, beta=50.0):
    """Soft objective using log-sum-exp for smoother landscape."""
    points = _polar_to_cartesian(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all triangle areas
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas.append(0.5 * np.abs(cross))
    
    if not areas:
        return 1e10
    
    areas = np.concatenate(areas)
    if np.min(areas) < 1e-12:
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    
    # Log-sum-exp for soft minimum
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
    
    return hull_area / soft_min_val


def _initial_params_regular():
    """Generate initial polar parameters for a regular polygon."""
    # Regular N-gon with circumradius R
    R = np.sqrt(2.0 / (N * np.sin(2.0 * np.pi / N)))
    
    radii = np.ones(N) * R
    dthetas = np.full(N - 1, 2.0 * np.pi / N)
    
    return np.concatenate([radii, dthetas])


def _local_refine(params, max_iter=50):
    """Local refinement using L-BFGS-B."""
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < _objective(params):
            return result.x
        return params
    except Exception:
        return params


def construct():
    start_time = time.time()
    
    # Define bounds for polar coordinates
    # Radii: [0.1, 3.0]
    # dthetas: [0.01, 2*pi/N * 3]
    min_dtheta = 0.01
    max_dtheta = 2.0 * np.pi / N * 3.0
    
    r_bounds = [(0.1, 3.0)] * N
    dtheta_bounds = [(min_dtheta, max_dtheta)] * (N - 1)
    bounds = r_bounds + dtheta_bounds
    
    # Initial guess: regular polygon
    x0 = _initial_params_regular()
    
    best_score = 0.0
    best_params = None
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0,
            seed=42,
            maxiter=300,
            tol=1e-10,
            popsize=12,
            polish=False,
            init='sobol',
            updating='immediate'
        )
        
        if result.x is not None:
            cand_pts = _polar_to_cartesian(result.x)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = result.x
    except Exception:
        pass
    
    # Local refinement
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        try:
            refined = _local_refine(best_params, max_iter=100)
            cand_pts = _polar_to_cartesian(refined)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = refined
        except Exception:
            pass
    
    # Try soft objective
    if time.time() - start_time < DEADLINE - 25 and best_params is not None:
        try:
            result2 = differential_evolution(
                lambda p: _objective_soft(p, beta=100.0),
                bounds=bounds,
                x0=x0,
                seed=43,
                maxiter=200,
                tol=1e-10,
                popsize=10,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if result2.x is not None:
                refined2 = _local_refine(result2.x, max_iter=50)
                cand_pts = _polar_to_cartesian(refined2)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = refined2
        except Exception:
            pass
    
    # Another DE run with different seed
    if time.time() - start_time < DEADLINE - 10:
        try:
            result3 = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0,
                seed=44,
                maxiter=200,
                tol=1e-10,
                popsize=8,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if result3.x is not None:
                refined3 = _local_refine(result3.x, max_iter=50)
                cand_pts = _polar_to_cartesian(refined3)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = refined3
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_params is None:
        best_params = x0
    
    return _polar_to_cartesian(best_params)
# EVOLVE-BLOCK-END
