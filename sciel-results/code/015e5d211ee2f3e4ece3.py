# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized computation."""
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
    
    # Vectorized triangle area computation
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # We compute all pairwise differences and use broadcasting
    # n=13, so C(13,3)=286 triangles - manageable with vectorization
    
    idx = np.arange(n)
    # Create all index triples
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    i_idx = i_idx.flatten()
    j_idx = j_idx.flatten()
    k_idx = k_idx.flatten()
    
    # Keep only i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Compute triangle areas
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Cross product: (pj-pi) x (pk-pi)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iters: int = 500, step_size: float = 0.05,
                  learning_rate: float = 0.85, min_step: float = 1e-7,
                  deadline: float = None) -> np.ndarray:
    """Perform local search with adaptive step size and multi-directional moves."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_step = step_size
    n = len(best_points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try moving each point in multiple directions
        for i in range(n):
            # Try 8 directions + 4 axis-aligned + diagonal
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1),
                (1, 0.5), (-1, -0.5), (0.5, 1), (-0.5, -1),
            ]
            
            for dx_dir, dy_dir in directions:
                for sign in [1, -1]:
                    dx = sign * current_step * dx_dir
                    dy = sign * current_step * dy_dir
                    
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
            # Slightly increase step if we found improvements
            current_step = min(current_step * 1.05, 0.2)
    
    return best_points


def _powell_refine(points: np.ndarray, deadline: float = None, max_iters: int = 100) -> np.ndarray:
    """Powell's method for unconstrained optimization (with bounds clamping)."""
    n = len(points)
    best_points = points.copy().flatten()
    best_score = _compute_score(best_points.reshape(n, 2))
    
    # Initial directions: unit vectors for each coordinate
    directions = []
    for d in range(2 * n):
        dir_vec = np.zeros(2 * n)
        dir_vec[d] = 1.0
        directions.append(dir_vec)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        old_score = best_score
        
        for d in range(len(directions)):
            # Find optimal step along direction d
            best_step = 0.0
            best_dir_score = best_score
            
            # Try several step sizes
            for step in [0.01, 0.02, 0.05, 0.1, 0.2]:
                for sign in [1, -1]:
                    candidate = best_points + sign * step * directions[d]
                    # Clamp to [0, 1]
                    candidate = np.clip(candidate, 0, 1)
                    
                    pts = candidate.reshape(n, 2)
                    if not np.all(np.isfinite(pts)):
                        continue
                    
                    score = _compute_score(pts)
                    if score > best_dir_score:
                        best_dir_score = score
                        best_step = sign * step
            
            # Update best_points with the best step found
            if best_step != 0:
                best_points = best_points + best_step * directions[d]
                best_points = np.clip(best_points, 0, 1)
                best_score = best_dir_score
        
        # Check for convergence
        if abs(best_score - old_score) < 1e-10:
            break
        
        # Add the most productive direction
        # Simplified: just keep cycling through directions
    
    return best_points.reshape(n, 2)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (20s buffer)
    
    best_points = None
    best_score = 0.0
    
    # Try multiple random initializations with different strategies
    for seed in range(100):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        rng = np.random.default_rng(seed=seed)
        
        # Different initialization strategies
        if seed % 3 == 0:
            # Random uniform
            points = rng.random((n, 2))
        elif seed % 3 == 1:
            # Slightly perturbed grid
            base_x = np.linspace(0.1, 0.9, 4)
            base_y = np.linspace(0.1, 0.9, 4)
            grid = []
            for x in base_x:
                for y in base_y:
                    grid.append([x, y])
            points = np.array(grid[:n])
            points += rng.normal(0, 0.05, (n, 2))
            points = np.clip(points, 0.05, 0.95)
        else:
            # Random with some structure
            points = rng.random((n, 2))
            # Add some spacing by sorting and adjusting
            points[:, 0] = np.sort(points[:, 0])
            points[:, 1] = np.sort(points[:, 1])[::-1]  # anti-diagonal tendency
        
        # Run local search
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        points = _local_search(points, max_iters=300, step_size=0.05,
                               learning_rate=0.85, min_step=1e-7,
                               deadline=deadline)
        
        # Powell refinement
        remaining = deadline - time.time()
        if remaining > 10:
            points = _powell_refine(points, deadline=deadline, max_iters=50)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # If we're doing well and time is running out, stop
        if best_score > 70 and elapsed > 300:
            break
    
    # If no valid points found, return a basic configuration
    if best_points is None:
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        best_points = np.array(grid_points[:13])
    
    return best_points


# EVOLVE-BLOCK-END
