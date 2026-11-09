# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
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
    
    # Vectorized triangle area calculation
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _fast_score(points: np.ndarray, precomputed_areas=None, precomputed_hull_area=None) -> float:
    """Fast scoring that can reuse precomputed values when only one point changes."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
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
    
    # Vectorized triangle area calculation
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _incremental_score(points: np.ndarray, changed_idx: int, old_areas: np.ndarray, old_A_hull: float) -> float:
    """
    Score a configuration where only points[changed_idx] has changed.
    old_areas: array of all 286 triangle areas from previous state
    old_A_hull: previous hull area (approximation - hull may change slightly)
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Recompute hull (it may change)
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Only triangles involving changed_idx need recomputation
    # Triangles: (changed_idx, j, k) for all j < k, j != changed_idx, k != changed_idx
    # Plus (i, changed_idx, k) for i < k, i != changed_idx, k != changed_idx
    # Plus (i, j, changed_idx) for i < j, i != changed_idx, j != changed_idx
    
    new_areas = old_areas.copy()
    
    # Find all triangles involving changed_idx
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    # Triangles where changed_idx is one of the vertices
    tri_mask = (i_sel == changed_idx) | (j_sel == changed_idx) | (k_sel == changed_idx)
    
    if np.any(tri_mask):
        tri_i = i_sel[tri_mask]
        tri_j = j_sel[tri_mask]
        tri_k = k_sel[tri_mask]
        
        p1 = points[tri_i]
        p2 = points[tri_j]
        p3 = points[tri_k]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        new_areas[tri_mask] = np.abs(cross) * 0.5
    
    min_area = np.min(new_areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iter=10000, initial_step=0.05, time_limit=None) -> np.ndarray:
    """Perform local search with incremental scoring to improve the point arrangement."""
    current = points.copy()
    n = current.shape[0]
    step_size = initial_step
    start_time = time.time()
    
    # Precompute initial areas
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    p1 = current[i_sel]
    p2 = current[j_sel]
    p3 = current[k_sel]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    current_areas = np.abs(cross) * 0.5
    
    try:
        hull = ConvexHull(current)
        current_A_hull = hull.volume
    except Exception:
        return current
    
    current_score = _score(current)
    
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
        
        # Use incremental scoring
        candidate_score = _incremental_score(candidate, idx, current_areas, current_A_hull)
        
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            
            # Update areas for next iteration
            p1 = current[i_sel]
            p2 = current[j_sel]
            p3 = current[k_sel]
            cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
            current_areas = np.abs(cross) * 0.5
            
            try:
                hull = ConvexHull(current)
                current_A_hull = hull.volume
            except Exception:
                pass
            
            # Slightly decrease step size to refine
            step_size = max(step_size * 0.99, 0.001)
        else:
            # Slightly increase step size to explore
            step_size = min(step_size * 1.01, 0.15)
    
    return current

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 610  # 20s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Grid-like
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.02, 0.98)
    
    # 2. Hexagonal packing
    hex_pts = np.array([
        [0.15, 0.15], [0.45, 0.15], [0.75, 0.15],
        [0.30, 0.38], [0.60, 0.38],
        [0.15, 0.61], [0.45, 0.61], [0.75, 0.61],
        [0.30, 0.84], [0.60, 0.84],
        [0.45, 0.05], [0.30, 0.55], [0.60, 0.55]
    ])
    
    # 3. Regular polygon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius = 0.4
    poly_pts = np.column_stack([0.5 + radius*np.cos(angles), 0.5 + radius*np.sin(angles)])
    poly_pts = np.vstack([poly_pts, [[0.5, 0.5]]])
    
    # 4. Random seeds with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    random_starts = [rng.random((13, 2)) for _ in range(8)]
    random_starts = [np.clip(p, 0.02, 0.98) for p in random_starts]
    
    starts = [grid, hex_pts, poly_pts] + random_starts
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Local search from best start with extended time
    if best_points is not None:
        improved = _local_search(best_points, max_iter=20000, initial_step=0.08, time_limit=deadline - 30)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Random restarts with local search
    while time.time() < deadline - 15:
        pts = rng.random((13, 2))
        pts = np.clip(pts, 0.02, 0.98)
        
        # Quick check
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Local search with limited time per restart
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
        restart_deadline = min(time.time() + min(15, remaining), deadline - 10)
        improved = _local_search(pts, max_iter=5000, initial_step=0.1, time_limit=restart_deadline)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    if best_points is None:
        best_points = np.random.default_rng(42).random((13, 2))
    
    return best_points

# EVOLVE-BLOCK-END
