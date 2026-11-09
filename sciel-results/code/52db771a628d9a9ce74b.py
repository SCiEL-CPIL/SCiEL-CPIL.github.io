# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _min_dist(points: np.ndarray) -> float:
    """Return the minimum pairwise distance."""
    dists = pdist(points)
    return np.min(dists)


def _local_optimize(points: np.ndarray, max_iters: int = 500, step_size: float = 0.05) -> np.ndarray:
    """Local optimization to maximize minimum pairwise distance."""
    points = points.copy()
    current_min_dist = _min_dist(points)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(points)):
            for attempt in range(20):
                # Random perturbation
                noise = np.random.randn(3) * step_size
                new_points = points.copy()
                new_points[i] += noise
                
                # Check if point is still finite
                if not np.all(np.isfinite(new_points[i])):
                    continue
                
                new_min_dist = _min_dist(new_points)
                
                if new_min_dist > current_min_dist:
                    points = new_points
                    current_min_dist = new_min_dist
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n = 14
    d = 3
    
    # Start with points on a unit sphere using a structured approach
    # Use icosahedron-like distribution with additional points
    
    # Initialize with random points on unit sphere
    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    # Try multiple random initializations
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random points on unit sphere
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization
        optimized = _local_optimize(points, max_iters=200, step_size=0.1)
        
        score = _evaluate(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
        
        # If we've found a very good solution, we can stop early
        if best_score > 0.4:
            break
    
    if best_points is None:
        # Fallback: random points
        points = np.random.randn(n, d)
        return points
    
    return best_points


# EVOLVE-BLOCK-END
