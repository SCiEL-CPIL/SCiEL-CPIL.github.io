# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize


def _score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _negative_score(points):
    """Negative of score for minimization."""
    return -_score(points)


def _powell_refine(points, start_time, deadline):
    """Refine points using Powell's method with time guard."""
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    try:
        result = minimize(
            obj, x0, method='Powell',
            options={'maxiter': 200, 'maxfev': 5000}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points


def _hexagonal_grid(n, scale=1.0):
    """Generate points on a hexagonal grid."""
    points = []
    dx = scale
    dy = scale * np.sqrt(3) / 2
    # Generate enough points
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            points.append([x, y])
    points = np.array(points)
    # Select n points closest to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _random_init(n, seed):
    """Random initialization with fixed seed."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, 2) * 2.0


def _circle_init(n, radius=1.0, angle_offset=0.0):
    """Points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _grid_init(n, scale=1.0):
    """Simple square grid initialization."""
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            points.append([i * scale, j * scale])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    best_points = None
    best_score = -1.0

    # Generate diverse initializations
    inits = []
    
    # Hexagonal grid
    inits.append(_hexagonal_grid(n, scale=1.0))
    inits.append(_hexagonal_grid(n, scale=0.8))
    inits.append(_hexagonal_grid(n, scale=1.2))
    
    # Grid
    inits.append(_grid_init(n, scale=1.0))
    inits.append(_grid_init(n, scale=0.9))
    
    # Circle
    inits.append(_circle_init(n, radius=1.0, angle_offset=0.0))
    inits.append(_circle_init(n, radius=1.0, angle_offset=np.pi / n))
    
    # Random inits with different seeds
    for seed in range(10):
        inits.append(_random_init(n, seed))
    
    # Evaluate initial scores
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    # Refine each initialization using Powell's method
    for init in inits:
        if time.time() >= deadline:
            break
        
        refined = _powell_refine(init, start_time, deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
    
    # Additional refinement on the best point found
    if best_points is not None and time.time() < deadline - 30:
        refined = _powell_refine(best_points, start_time, deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
    
    # Final check
    if best_points is None:
        # Fallback to a simple configuration
        best_points = _hexagonal_grid(n, scale=1.0)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _hexagonal_grid(n, scale=1.0)
    
    return best_points


# EVOLVE-BLOCK-END
