# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # 5s safety margin before 600s kill


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


def _soft_min_triangle_area(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This gives a differentiable surrogate for optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, shift by min
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 0.0
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return float(soft_min)


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


def _soft_score(points, beta=50.0):
    """
    Compute a smooth approximation of the score using soft-min triangle area.
    """
    a_min_soft = _soft_min_triangle_area(points, beta)
    a_hull = _hull_area(points)
    if a_hull <= 1e-15:
        return 0.0
    return a_min_soft / a_hull


def _objective(x):
    """
    Objective function for optimization: negative score.
    """
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _soft_objective(x, beta=50.0):
    """
    Objective function for optimization: negative soft score (smoother landscape).
    """
    points = x.reshape(-1, 2)
    s = _soft_score(points, beta)
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


def _regular_polygon_with_center(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points: regular (n-1)-gon plus center point.
    For n=13: 12-gon + center = 13 points.
    """
    if n <= 1:
        return np.array([[center[0], center[1]]])
    
    outer_n = n - 1
    angles = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
    outer_points = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    center_pt = np.array([[center[0], center[1]]])
    return np.vstack([center_pt, outer_points])


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
    
    # Generate diverse initializations - focus on structured topologies
    initializations = []
    
    # 1. Regular 13-gon (strong baseline from inspiration 1)
    initializations.append(_regular_polygon(n, radius=1.0))
    
    # 2. Regular 12-gon + center
    initializations.append(_regular_polygon_with_center(n, radius=1.0))
    
    # 3. Concentric rings: 1 center + 5 inner + 7 outer
    def _concentric_rings(n):
        pts = [[0.0, 0.0]]
        # Inner ring: 5 points at radius 0.5
        for i in range(5):
            angle = i * 2 * np.pi / 5
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        # Outer ring: 7 points at radius 1.0
        for i in range(7):
            angle = i * 2 * np.pi / 7 + np.pi / 7
            pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        return np.array(pts[:n])
    initializations.append(_concentric_rings(n))
    
    # 4. Fibonacci spiral (good space-filling)
    initializations.append(_circle_interior(n, radius=1.0))
    
    # 5. Perturbed regular polygon (breaks symmetry slightly)
    for seed in [42, 123, 456, 789, 1011]:
        rng = np.random.default_rng(seed)
        base = _regular_polygon(n, radius=1.0)
        noise = rng.normal(0, 0.05, (n, 2))
        initializations.append(base + noise)
    
    # 6. Random points in unit circle
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        r = np.sqrt(rng.random(n))
        theta = 2 * np.pi * rng.random(n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(pts)
    
    # Evaluate all initializations
    scored_inits = []
    for init in initializations:
        if len(init) != n:
            continue
        try:
            s = _score(init)
        except Exception:
            continue
        scored_inits.append((s, init))
    
    if not scored_inits:
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
        return best_points
    
    scored_inits.sort(key=lambda x: -x[0])
    best_score = scored_inits[0][0]
    best_points = scored_inits[0][1].copy()
    
    # Phase 1: Coarse optimization with soft objective (smoother landscape)
    # Optimize top 4 initializations
    num_coarse = min(4, len(scored_inits))
    
    for rank in range(num_coarse):
        if time.time() > _DEADLINE - 120:
            break
        
        s_init, init = scored_inits[rank]
        x0 = init.flatten()
        
        try:
            # Use soft objective for smoother landscape
            result = minimize(
                lambda x: _soft_objective(x, beta=30.0),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            continue
    
    # Phase 2: Fine optimization with exact objective
    # Start from current best and a few other good starts
    fine_starts = [best_points.copy()]
    for s_init, init in scored_inits[1:min(3, len(scored_inits))]:
        fine_starts.append(init.copy())
    
    for i, init in enumerate(fine_starts):
        if time.time() > _DEADLINE - 60:
            break
        
        x0 = init.flatten()
        
        try:
            # Powell for local refinement
            result = minimize(
                _objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 500,
                    'maxfev': 20000,
                    'ftol': 1e-14,
                    'xtol': 1e-14
                }
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            continue
    
    # Phase 3: If time remains, do targeted perturbations of best solution
    if time.time() < _DEADLINE - 30:
        rng = np.random.default_rng(12345)
        for trial in range(20):
            if time.time() > _DEADLINE - 30:
                break
            
            # Small perturbation of best solution
            noise_scale = 0.01 * (1 - trial / 20.0)
            perturbed = best_points + rng.normal(0, noise_scale, (n, 2))
            
            try:
                s_pert = _score(perturbed)
                if s_pert > best_score:
                    best_score = s_pert
                    best_points = perturbed.copy()
                else:
                    # Try to refine the perturbed point
                    x0 = perturbed.flatten()
                    result = minimize(
                        _objective,
                        x0,
                        method='Powell',
                        options={
                            'maxiter': 100,
                            'maxfev': 3000,
                            'ftol': 1e-13,
                            'xtol': 1e-13
                        }
                    )
                    opt_points = result.x.reshape(-1, 2)
                    opt_score = _score(opt_points)
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
