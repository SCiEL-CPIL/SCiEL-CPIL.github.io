"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    # Use Andrew's monotone chain algorithm for convex hull
    points = np.array(points)
    points = np.unique(points, axis=0)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Sort by x, then by y
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate lower and upper hull, removing duplicate endpoints
    hull = np.vstack([lower[:-1], upper[:-1]])
    m = len(hull)
    if m < 3:
        return 0.0
    
    # Compute area using shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective(x):
    """Objective function for optimization (minimize negative score)."""
    points = x.reshape(-1, 2)
    return -_score(points)


def _initial_guess():
    """Generate an initial guess using a grid-like pattern on a circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Place points on a circle with some radial variation
    radii = np.ones(N)
    # Make some points inner to create more triangles
    radii[::3] = 0.5
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points.flatten()


def construct():
    """Construct 13 points to maximize the minimum triangle area / convex hull area."""
    # Start with a good initial guess
    x0 = _initial_guess()
    
    # Use differential evolution for global optimization
    # Bounds: points in [0, 1] x [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    best_result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        polish=True,
        init='sobol'
    )
    
    best_points = best_result.x.reshape(-1, 2)
    best_score = _score(best_points)
    
    # Also try local refinement from multiple starting points
    start_points = [x0]
    # Add a few more diverse starting points
    rng = np.random.default_rng(42)
    for _ in range(3):
        start_points.append(rng.uniform(0, 1, size=2 * N))
    
    for sp in start_points:
        res = minimize(_objective, sp, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
        pts = res.x.reshape(-1, 2)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Normalize: the score is already min_tri / hull_area, so we just return the points
    return best_points
# EVOLVE-BLOCK-END
