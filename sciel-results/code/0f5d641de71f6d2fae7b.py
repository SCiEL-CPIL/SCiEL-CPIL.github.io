"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def _convex_hull_area_fast(points):
    """Fast convex hull area using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Sort by x, then y
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _min_triangle_area_fast(points):
    """Fast min triangle area using vectorized computation."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area


def _score_from_params(params):
    """
    Compute score from parameterized points.
    params: array of shape (2*N,) where first N are angles, next N are radii.
    Points are placed at (r_i * cos(theta_i), r_i * sin(theta_i)).
    """
    angles = params[:N]
    radii = params[N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.01)
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    hull_area = _convex_hull_area_fast(pts)
    if hull_area < 1e-10:
        return 0.0
    
    min_tri = _min_triangle_area_fast(pts)
    return min_tri / hull_area


def _negative_score(params):
    """Negative score for minimization."""
    return -_score_from_params(params)


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    New approach: Differential evolution over angular and radial parameters.
    Points are parameterized as (r_i * cos(theta_i), r_i * sin(theta_i)).
    This keeps points in a star-shaped region and the optimization landscape
    is smoother than raw coordinate optimization.
    """
    start_time = time.time()
    time_limit = 100  # Leave buffer for overhead
    
    # Bounds: angles in [0, 2*pi), radii in [0.1, 1.5]
    angles_bounds = [(0, 2 * np.pi)] * N
    radii_bounds = [(0.1, 1.5)] * N
    bounds = angles_bounds + radii_bounds
    
    # Initial population: regular polygon + variations
    def make_init_pop():
        pop = []
        # Regular polygon
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base_radii = np.ones(N)
        pop.append(np.concatenate([base_angles, base_radii]))
        
        # Perturbed regular
        rng = np.random.RandomState(42)
        for _ in range(19):
            angles = base_angles + rng.uniform(-0.1, 0.1, N)
            radii = 1.0 + rng.uniform(-0.2, 0.2, N)
            pop.append(np.concatenate([angles, radii]))
        
        # Two-ring structure
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        inner_angles = outer_angles + np.pi / 7
        angles2 = np.concatenate([outer_angles, inner_angles])
        radii2 = np.concatenate([np.ones(7), np.full(7, 0.5)])
        pop.append(np.concatenate([angles2, radii2]))
        
        # Random
        for _ in range(10):
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.3, 1.2, N)
            pop.append(np.concatenate([angles, radii]))
        
        return np.array(pop)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _negative_score,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            init=make_init_pop,
            updating='immediate',
            polish=True
        )
        
        best_params = result.x
        best_score = -result.fun
        
        # Check time
        if time.time() - start_time > time_limit:
            # Fallback to best so far
            pass
        
    except Exception as e:
        best_params = None
        best_score = -1.0
    
    # If DE failed or didn't improve, try local refinement on best params
    if best_params is not None:
        # Evaluate and refine with Nelder-Mead if time permits
        if time.time() - start_time < time_limit - 10:
            from scipy.optimize import minimize
            try:
                refined = minimize(_negative_score, best_params, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                if -refined.fun > best_score:
                    best_params = refined.x
                    best_score = -refined.fun
            except Exception:
                pass
    
    # Convert params to points
    if best_params is not None:
        angles = best_params[:N]
        radii = np.maximum(best_params[N:], 0.01)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
# EVOLVE-BLOCK-END
