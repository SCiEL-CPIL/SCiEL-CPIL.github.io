"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on sorted points
        # Find convex hull manually
        n = len(points)
        if n < 3:
            return 0.0
        # Sort by x then y
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
                lower.pop()
            lower.append(p)
        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
                upper.pop()
            upper.append(p)
        # Concatenate
        hull_pts = lower[:-1] + upper[:-1]
        # Shoelace formula
        m = len(hull_pts)
        area = 0.0
        for i in range(m):
            j = (i + 1) % m
            area += hull_pts[i][0] * hull_pts[j][1]
            area -= hull_pts[j][0] * hull_pts[i][1]
        return abs(area) / 2.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-15:
        return 0.0
    return min_area / hull_area


def _negative_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(N, 2)
    return -_score(points)


def _regular_polygon_init(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _concentric_rings_init(n, n_rings=2):
    """Generate points on concentric rings."""
    points = []
    # Outer ring
    n_outer = n // 2
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points.extend([(np.cos(a), np.sin(a)) for a in angles_outer])
    # Inner ring
    n_inner = n - n_outer
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    r_inner = 0.5
    points.extend([(r_inner * np.cos(a), r_inner * np.sin(a)) for a in angles_inner])
    return np.array(points[:n])


def _grid_init(n):
    """Generate points on a grid pattern."""
    # For 14 points, use a 4x4 grid with 2 removed
    points = []
    for i in range(4):
        for j in range(4):
            points.append([(i - 1.5) * 0.8, (j - 1.5) * 0.8])
    # Remove 2 points to get 14
    points = points[:n]
    return np.array(points)


def _random_init(n, seed):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1, 1, size=(n, 2))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_regular_polygon_init(N))
    inits.append(_concentric_rings_init(N))
    inits.append(_grid_init(N))
    
    # Add several random inits
    for seed in range(10):
        inits.append(_random_init(N, seed))
    
    # Also try scaled versions
    for scale in [0.5, 1.5, 2.0]:
        inits.append(_regular_polygon_init(N, radius=scale))
    
    for init_points in inits:
        if time.time() > deadline - 5.0:
            break
        
        # Evaluate initial score
        init_score = _score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                _negative_score,
                init_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10
                }
            )
            
            opt_points = result.x.reshape(N, 2)
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception as e:
            continue
    
    # Final polish: run a few more iterations from the best point
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            polished_points = result.x.reshape(N, 2)
            polished_score = _score(polished_points)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
