# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas efficiently
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
                if min_area == 0:
                    return 0.0
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iters: int = 1000, step_size: float = 0.1,
                  learning_rate: float = 0.95, min_step: float = 1e-6,
                  deadline: float = None) -> np.ndarray:
    """Perform local search to improve the point configuration."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_step = step_size
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best_points)):
            for dx in [-current_step, 0, current_step]:
                for dy in [-current_step, 0, current_step]:
                    if dx == 0 and dy == 0:
                        continue
                    candidate = best_points.copy()
                    candidate[i, 0] += dx
                    candidate[i, 1] += dy
                    
                    # Keep points in [0, 1]^2
                    if np.any(candidate < 0) or np.any(candidate > 1):
                        continue
                    
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
        
        if not improved:
            current_step *= learning_rate
            if current_step < min_step:
                break
        else:
            current_step = max(current_step, min_step)
    
    return best_points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600  # Safety margin
    
    # Start with a grid-like initialization
    best_points = None
    best_score = 0.0
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() > deadline - 30:  # Stop if approaching deadline
            break
        
        rng = np.random.default_rng(seed=seed)
        points = rng.random((n, 2))
        
        # Run local search
        points = _local_search(points, max_iters=200, step_size=0.1,
                               learning_rate=0.9, min_step=1e-5,
                               deadline=deadline)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If no valid points found, return a basic configuration
    if best_points is None:
        # Use a simple grid pattern
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        # Take first 13 points
        best_points = np.array(grid_points[:13])
    
    return best_points


# EVOLVE-BLOCK-END
