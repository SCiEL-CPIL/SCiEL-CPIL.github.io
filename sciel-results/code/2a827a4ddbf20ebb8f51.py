"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 110.0  # safety margin before 120s kill


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(points)
    if n < 3:
        return 0.0
    # Vectorized computation of all triangle areas
    i, j, k = np.indices((n, n, n))
    # Only keep i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_a


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _initial_grid(n):
    """Create a well-distributed initial configuration on a unit square."""
    # Use a hexagonal-like packing for better distribution
    points = []
    cols = 4
    rows = 3
    # Arrange in a staggered grid
    for r in range(rows):
        for c in range(cols):
            if len(points) >= n:
                break
            x = c / (cols - 1)
            y = r / (rows - 1)
            if r % 2 == 1:
                x += 0.5 / (cols - 1)
            points.append([x, y])
    # Add remaining points
    while len(points) < n:
        points.append([0.5, 0.5])
    return np.array(points[:n])


def _optimize_from(start_points, deadline, rng):
    """Optimize from a starting configuration using L-BFGS-B."""
    best_points = start_points.copy()
    best_score = _score(best_points)
    
    def objective(x):
        points = x.reshape(-1, 2)
        return _negative_score(points)
    
    x0 = start_points.flatten()
    
    # Try L-BFGS-B with bounds (keep points in a reasonable region)
    bounds = [(-1, 2)] * (2 * N)
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < _negative_score(x0):
            cand = result.x.reshape(-1, 2)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_points = cand
                best_score = cand_score
    except Exception:
        pass
    
    return best_points, best_score


def construct():
    """Construct 13 points maximizing min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = None
    best_score = 0.0
    
    # Initial configuration: staggered grid
    grid_points = _initial_grid(N)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if attempt == 0:
            start_pts = grid_points.copy()
        else:
            # Perturb the grid
            start_pts = grid_points + rng.normal(0, 0.05, size=(N, 2))
        
        pts, score = _optimize_from(start_pts, deadline, rng)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # If we have good time, also try optimizing from the best found so far
        if time.time() > deadline - 15.0:
            break
    
    # Final polish: one more optimization from the best point
    if best_points is not None and time.time() < deadline - 5.0:
        pts, score = _optimize_from(best_points, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return the grid
        return grid_points
    
    return best_points
