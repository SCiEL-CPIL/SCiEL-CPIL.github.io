# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 570.0  # 30s safety margin before 600s kill


def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))


def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """
    Compute the score: A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-15:
        return 0.0
    return a_min / a_hull


def _objective(x):
    """
    Objective function for optimization: negative score.
    """
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points on a regular polygon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return points


def _circle_interior(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points inside a circle (Fibonacci-like spiral for even distribution).
    """
    # Use golden angle spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    theta = i * golden_angle
    points = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta)
    ])
    return points


def _random_points(n, seed=42):
    """
    Generate random points in a unit square.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _perturbed_circle(n, radius=1.0, center=(0.0, 0.0), noise=0.1, seed=42):
    """
    Generate points on a circle with random perturbation.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-noise, noise, n)
    radii = radius * (1 + rng.uniform(-0.2, 0.2, n))
    points = np.column_stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles)
    ])
    return points


def _grid_points(n):
    """
    Generate points on a grid-like arrangement.
    For n=13, use a 4x4 grid minus corners, or similar.
    """
    # 4x4 grid = 16 points, remove 3 to get 13
    # Or use a hexagonal-like pattern
    # Let's try a structured approach: rows of 3,3,4,3
    rows = [3, 3, 4, 3]
    points = []
    y = 0
    for count in rows:
        x_start = -(count - 1) / 2.0
        for i in range(count):
            points.append([x_start + i, y])
        y += 1
    points = np.array(points)
    # Center it
    points -= points.mean(axis=0)
    return points


def _hexagonal_points(n):
    """
    Generate points in a hexagonal lattice pattern.
    """
    # Hexagonal packing: center + 6 neighbors + 12 next ring = 1+6+12=19
    # For 13: center + 6 + 6 from next ring
    points = [[0, 0]]  # center
    
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    
    # Second ring partial (6 more points)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    
    points = np.array(points[:n])
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon(n, radius=1.0))
    
    # 2. Circle interior (Fibonacci spiral)
    initializations.append(_circle_interior(n, radius=1.0))
    
    # 3. Random points
    for seed in [42, 123, 456, 789, 1011]:
        initializations.append(_random_points(n, seed=seed))
    
    # 4. Perturbed circle
    for seed in [42, 123, 456]:
        initializations.append(_perturbed_circle(n, radius=1.0, noise=0.1, seed=seed))
    
    # 5. Grid-like
    initializations.append(_grid_points(n))
    
    # 6. Hexagonal
    initializations.append(_hexagonal_points(n))
    
    # Evaluate initializations
    best_points = None
    best_score = -np.inf
    
    for init in initializations:
        if len(init) != n:
            continue
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    if best_points is None:
        # Fallback to random
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
        best_score = _score(best_points)
    
    # Local optimization from best initializations
    # Sort initializations by score and optimize top ones
    scored_inits = []
    for init in initializations:
        if len(init) != n:
            continue
        s = _score(init)
        scored_inits.append((s, init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Optimize top 5 initializations
    num_optimize = min(5, len(scored_inits))
    
    for rank in range(num_optimize):
        if time.time() > _DEADLINE:
            break
        
        s_init, init = scored_inits[rank]
        
        # Flatten for optimization
        x0 = init.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': 5000,
                    'ftol': 1e-12,
                    'xtol': 1e-12
                }
            )
            
            if result.success or result.fun < -best_score:
                opt_points = result.x.reshape(-1, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            continue
    
    # If we still have time, do a few more random restarts with optimization
    while time.time() < _DEADLINE - 30:
        seed = np.random.randint(0, 1000000)
        init = _random_points(n, seed=seed)
        
        # Quick evaluation
        s_init = _score(init)
        if s_init < best_score * 0.5:
            continue
        
        x0 = init.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 100,
                    'maxfev': 3000,
                    'ftol': 1e-12,
                    'xtol': 1e-12
                }
            )
            
            if result.success or result.fun < -best_score:
                opt_points = result.x.reshape(-1, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            continue
    
    # Final validation
    if best_points is None or best_points.shape != (n, 2):
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
    
    return best_points


# EVOLVE-BLOCK-END
