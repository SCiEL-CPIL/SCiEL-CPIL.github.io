"""Program for Heilbronn convex regions n=14 using polar coordinates and differential evolution."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 110.0  # Safety margin before 120s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
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


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _polar_to_cartesian(params):
    """Convert polar parameters (r_i, theta_i) to Cartesian, enforcing sorted angles."""
    n = len(params) // 2
    radii = params[:n]
    angles = params[n:]
    # Sort by angle and reorder radii accordingly
    order = np.argsort(angles)
    sorted_angles = angles[order]
    sorted_radii = radii[order]
    x = sorted_radii * np.cos(sorted_angles)
    y = sorted_radii * np.sin(sorted_angles)
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for minimization: negative of score."""
    points = _polar_to_cartesian(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-14:
        return 1e10
    return -min_area / hull_a


def _soft_objective(params, beta=100.0):
    """Soft objective using log-sum-exp for smoother landscape."""
    points = _polar_to_cartesian(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
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


def _initial_polar(n):
    """Generate initial polar parameters for a regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    radii = np.ones(n)
    # Scale to unit area: for regular n-gon with circumradius R,
    # area = (n/2) * R^2 * sin(2*pi/n)
    # We want area = 1, so R = sqrt(2/(n*sin(2*pi/n)))
    R = np.sqrt(2.0 / (n * np.sin(2.0 * np.pi / n)))
    radii *= R
    return np.concatenate([radii, angles])


def construct():
    start_time = time.time()
    
    # Define bounds for polar coordinates
    # Radii: [0, 5] (generous upper bound)
    # Angles: [0, 2*pi)
    r_bounds = [(0.0, 5.0)] * N
    theta_bounds = [(0.0, 2.0 * np.pi)] * N
    bounds = r_bounds + theta_bounds
    
    # Initial guess: regular polygon
    x0_polar = _initial_polar(N)
    
    best_score = 0.0
    best_pts = None
    
    # Run differential evolution with the hard objective
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0_polar,
            seed=42,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            polish=False,
            init='sobol',
            updating='immediate'
        )
        
        if result.x is not None:
            cand = _polar_to_cartesian(result.x)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
    except Exception:
        pass
    
    # If time remains, try with soft objective for smoother optimization
    if time.time() - start_time < DEADLINE - 20 and best_pts is not None:
        try:
            # Convert best Cartesian back to polar for continuation
            # Actually, let's just run DE again with soft objective from scratch or near best
            result2 = differential_evolution(
                lambda p: _soft_objective(p, beta=150.0),
                bounds=bounds,
                x0=x0_polar,
                seed=43,
                maxiter=100,
                tol=1e-8,
                popsize=10,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if result2.x is not None:
                cand = _polar_to_cartesian(result2.x)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Final polish: run DE on the hard objective starting near the best found
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            # Create a small neighborhood search around the best solution
            # We'll do a local DE with tighter bounds around the best polar params
            # But we don't have the exact polar params that gave best_pts, so let's
            # just run one more DE with a different seed
            result3 = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0_polar,
                seed=44,
                maxiter=100,
                tol=1e-9,
                popsize=8,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if result3.x is not None:
                cand = _polar_to_cartesian(result3.x)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_pts is None:
        best_pts = _polar_to_cartesian(x0_polar)
    
    return best_pts
# EVOLVE-BLOCK-END
