# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

# Precompute all C(13,3) = 286 triangle index combinations
_TRIS = np.array(list(itertools.combinations(range(13), 3)), dtype=np.int32)

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using precomputed indices."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
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
    
    # Vectorized triangle area calculation using precomputed indices
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |cross product|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _soft_min_score(points: np.ndarray, beta: float = 50.0) -> float:
    """Smooth surrogate: soft-min of triangle areas / hull area.
    Higher is better. Uses log-sum-exp for numerical stability."""
    n = points.shape[0]
    if n < 3:
        return -np.inf
    
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -np.inf
    
    if A_hull <= 0:
        return -np.inf
    
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    # Soft-min: -log-sum-exp(-areas/beta) * beta / A_hull
    # This approximates min(areas)/A_hull as beta -> 0
    min_val = np.min(areas)
    if min_val < 1e-15:
        return -np.inf
    
    # For numerical stability, use: soft_min = beta * (min_val - log(sum(exp((min_val - areas)/beta))))
    scaled = (min_val - areas) / beta  # <= 0
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = min_val - beta * log_sum
    
    if soft_min <= 0:
        return -np.inf
        
    return soft_min / A_hull


def _negative_soft_min(points_flat: np.ndarray, beta: float = 50.0) -> float:
    """Objective for minimizer: negative soft-min score."""
    points = points_flat.reshape(13, 2)
    s = _soft_min_score(points, beta)
    if np.isneginf(s) or np.isnan(s):
        return 1e6
    return -s


def _local_search(points: np.ndarray, max_iter=20000, initial_step=0.05, time_limit=None) -> np.ndarray:
    """Perform local search to improve the point arrangement."""
    current = points.copy()
    current_score = _score(current)
    n = current.shape[0]
    step_size = initial_step
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
            
        # Pick a random point to move
        idx = np.random.randint(n)
        # Generate a small random perturbation
        delta = np.random.normal(0, step_size, 2)
        new_point = current[idx] + delta
        
        # Keep points within [0, 1]^2
        if new_point[0] < 0 or new_point[0] > 1 or new_point[1] < 0 or new_point[1] > 1:
            continue
        
        candidate = current.copy()
        candidate[idx] = new_point
        candidate_score = _score(candidate)
        
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            step_size = max(step_size * 0.97, 0.001)
        else:
            step_size = min(step_size * 1.03, 0.15)
    
    return current


def _gradient_optimize(points: np.ndarray, time_limit: float, beta_start: float = 200.0, beta_end: float = 10.0) -> np.ndarray:
    """Use L-BFGS-B with smooth soft-min objective for gradient-based optimization."""
    x0 = points.flatten()
    n_params = x0.shape[0]
    
    # Anneal beta from high to low for smooth -> sharp transition
    betas = [beta_start, beta_start * 0.5, beta_start * 0.25, beta_start * 0.1, beta_end]
    
    best_x = x0.copy()
    best_score = _score(points)
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        res = minimize(
            _negative_soft_min,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * n_params,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < -1e-6:
            candidate = res.x.reshape(13, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        
        x0 = best_x
    
    return best_x.reshape(13, 2)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s kill
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Grid
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.01, 0.99)
    
    # 2. Hexagonal
    hex_pts = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1],
        [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
        [0.2, 0.7], [0.4, 0.7]
    ])
    
    # 3. Centered hexagonal pattern
    cx, cy = 0.5, 0.5
    r1 = 0.15
    r2 = 0.35
    hex_center = np.array([[cx, cy]])
    hex_inner = np.array([[cx + r1 * np.cos(2 * np.pi * i / 6), 
                           cy + r1 * np.sin(2 * np.pi * i / 6)] for i in range(6)])
    hex_outer = np.array([[cx + r2 * np.cos(2 * np.pi * (i + 0.5) / 6), 
                           cy + r2 * np.sin(2 * np.pi * (i + 0.5) / 6)] for i in range(6)])
    hex_sym = np.vstack([hex_center, hex_inner, hex_outer])
    hex_sym = np.clip(hex_sym, 0.01, 0.99)
    
    # 4. Random seeds
    rng = np.random.default_rng(42)
    random_starts = [rng.random((13, 2)) for _ in range(10)]
    random_starts = [np.clip(p, 0.01, 0.99) for p in random_starts]
    
    starts = [grid, hex_pts, hex_sym] + random_starts
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 1: Local search from best start
    if best_points is not None:
        ls_deadline = min(deadline - 120, start_time + 60)
        improved = _local_search(best_points, max_iter=5000, initial_step=0.05, time_limit=ls_deadline)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Phase 2: Gradient optimization from best point
    if time.time() < deadline - 60:
        grad_deadline = min(deadline - 30, time.time() + 80)
        grad_result = _gradient_optimize(best_points, grad_deadline)
        grad_score = _score(grad_result)
        if grad_score > best_score:
            best_score = grad_score
            best_points = grad_result.copy()
    
    # Phase 3: Random restarts with local search + gradient
    while time.time() < deadline - 40:
        pts = rng.random((13, 2))
        pts = np.clip(pts, 0.01, 0.99)
        
        # Quick check
        score = _score(pts)
        
        # Local search with limited time
        restart_deadline = min(time.time() + 15, deadline - 20)
        improved = _local_search(pts, max_iter=2000, initial_step=0.1, time_limit=restart_deadline)
        improved_score = _score(improved)
        
        # Try gradient from local search result
        if time.time() < deadline - 10 and improved_score > best_score * 0.9:
            grad_deadline = min(time.time() + 10, deadline - 5)
            grad_result = _gradient_optimize(improved, grad_deadline)
            grad_score = _score(grad_result)
            if grad_score > improved_score:
                improved_score = grad_score
                improved = grad_result
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        # Very fine local search
        final_deadline = deadline - 2
        polished = _local_search(best_points, max_iter=10000, initial_step=0.005, time_limit=final_deadline)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
    
    if best_points is None:
        best_points = np.random.default_rng(42).random((13, 2))
    
    return best_points

# EVOLVE-BLOCK-END
