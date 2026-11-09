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
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas and find the minimum
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        if area < min_area:
            min_area = area
    
    return min_area / A_hull


def _local_search(points: np.ndarray, time_budget: float, rng: np.random.Generator) -> np.ndarray:
    """Perform local search to improve the point configuration."""
    n = points.shape[0]
    best_points = points.copy()
    best_score = _compute_score(best_points)
    start_time = time.time()
    
    # Initial perturbation sizes
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_budget:
        # Try moving each point with random perturbations
        improved = False
        for _ in range(20):  # Number of perturbation attempts per iteration
            if time.time() - start_time >= time_budget:
                break
            
            i = rng.integers(0, n)
            dx = rng.normal(0, step_size)
            dy = rng.normal(0, step_size)
            
            new_points = best_points.copy()
            new_points[i] = [new_points[i, 0] + dx, new_points[i, 1] + dy]
            
            # Keep points in a reasonable range
            new_points = np.clip(new_points, -1, 2)
            
            score = _compute_score(new_points)
            if score > best_score:
                best_points = new_points
                best_score = score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
        
        # Decay step size if no improvement
        if no_improve_count > max_no_improve:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 1e-6:
                break
    
    return best_points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial configurations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        # Generate initial configuration
        if restart == 0:
            # Regular polygon on a circle
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 1:
            # Grid-like configuration
            points = np.random.default_rng(seed=restart * 100 + 1).random((n, 2))
        else:
            # Random with some structure
            seed = restart * 1000 + 7
            rng_init = np.random.default_rng(seed=seed)
            points = rng_init.random((n, 2))
            # Normalize to unit square
            points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-10)
        
        # Run local search with time budget
        time_remaining = min(25, deadline - time.time())
        if time_remaining <= 0:
            break
        
        improved_points = _local_search(points, time_remaining, rng)
        
        score = _compute_score(improved_points)
        if score > best_score:
            best_score = score
            best_points = improved_points
    
    if best_points is None:
        # Fallback: return regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
