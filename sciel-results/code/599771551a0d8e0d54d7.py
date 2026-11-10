# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

# Pre-compute all triangle index triples for n=13
_N = 13
_TRIPLE_IDX = np.array(list(itertools.combinations(range(_N), 3)))
_TRIPLE_I = _TRIPLE_IDX[:, 0]
_TRIPLE_J = _TRIPLE_IDX[:, 1]
_TRIPLE_K = _TRIPLE_IDX[:, 2]


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using pre-computed indices."""
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
    
    pi = points[_TRIPLE_I]
    pj = points[_TRIPLE_J]
    pk = points[_TRIPLE_K]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iters: int = 200, step_size: float = 0.03,
                  learning_rate: float = 0.9, min_step: float = 1e-6,
                  deadline: float = None) -> np.ndarray:
    """Perform local search with adaptive step size, using fewer but smarter directions."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_step = step_size
    n = len(best_points)
    
    # Use fewer, well-chosen directions to reduce evaluations per iteration
    directions = np.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [0.707, 0.707], [-0.707, -0.707], [0.707, -0.707], [-0.707, 0.707],
    ])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            for dx_dir, dy_dir in directions:
                for sign in [1, -1]:
                    dx = sign * current_step * dx_dir
                    dy = sign * current_step * dy_dir
                    
                    candidate = best_points.copy()
                    candidate[i, 0] += dx
                    candidate[i, 1] += dy
                    
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
            current_step = min(current_step * 1.02, 0.15)
    
    return best_points


def _powell_refine(points: np.ndarray, deadline: float = None, max_iters: int = 50) -> np.ndarray:
    """Powell's method with proper line search along each coordinate direction."""
    n = len(points)
    best_points = points.copy().flatten()
    best_score = _compute_score(best_points.reshape(n, 2))
    
    # Initial directions: unit vectors for each coordinate
    directions = [np.zeros(2 * n) for _ in range(2 * n)]
    for d in range(2 * n):
        directions[d][d] = 1.0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        old_score = best_score
        
        for d in range(len(directions)):
            # Line search: golden-section-like with a few probes
            best_step = 0.0
            best_dir_score = best_score
            
            # Probe a range of steps
            for step in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
                for sign in [1, -1]:
                    candidate = best_points + sign * step * directions[d]
                    candidate = np.clip(candidate, 0, 1)
                    
                    pts = candidate.reshape(n, 2)
                    score = _compute_score(pts)
                    if score > best_dir_score:
                        best_dir_score = score
                        best_step = sign * step
            
            if best_step != 0:
                best_points = best_points + best_step * directions[d]
                best_points = np.clip(best_points, 0, 1)
                best_score = best_dir_score
        
        if abs(best_score - old_score) < 1e-12:
            break
    
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
    deadline = start_time + 570  # Safety margin (60s buffer)
    
    best_points = None
    best_score = 0.0
    
    def _regular_polygon(n_pts, radius=0.45, center=(0.5, 0.5)):
        """Generate a regular n-gon centered at center with given radius."""
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])
    
    # Generate diverse initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(n))
    
    # 2. Regular 13-gon with different rotation
    initializations.append(_regular_polygon(n, radius=0.45, center=(0.5, 0.5)) * 1.0)
    initializations[1] = np.column_stack([
        0.5 + 0.45 * np.cos(np.linspace(np.pi/13, 2*np.pi + np.pi/13, n, endpoint=False)),
        0.5 + 0.45 * np.sin(np.linspace(np.pi/13, 2*np.pi + np.pi/13, n, endpoint=False))
    ])
    
    # 3. Perturbed regular polygon
    rng = np.random.default_rng(42)
    initializations.append(_regular_polygon(n) + rng.normal(0, 0.03, (n, 2)))
    
    # 4. Perturbed grid (4x4 minus one corner)
    base_x = np.linspace(0.1, 0.9, 4)
    base_y = np.linspace(0.1, 0.9, 4)
    grid = []
    for x in base_x:
        for y in base_y:
            grid.append([x, y])
    initializations.append(np.array(grid[:n]))
    
    # 5. Random uniform
    rng2 = np.random.default_rng(123)
    initializations.append(rng2.random((n, 2)) * 0.8 + 0.1)
    
    # 6. Another perturbed polygon with different radius
    initializations.append(_regular_polygon(n, radius=0.4))
    
    # 7. Two concentric rings (5 + 8)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([0.5 + 0.45*np.cos(angles_outer), 0.5 + 0.45*np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner = np.column_stack([0.5 + 0.2*np.cos(angles_inner), 0.5 + 0.2*np.sin(angles_inner)])
    initializations.append(np.vstack([outer, inner]))
    
    # 8. Another random with structure
    rng3 = np.random.default_rng(777)
    pts = rng3.random((n, 2)) * 0.8 + 0.1
    pts[:, 0] = np.sort(pts[:, 0])
    initializations.append(pts)
    
    # Run optimization on each initialization
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        points = init_points.copy()
        
        # Local search
        points = _local_search(points, max_iters=150, step_size=0.03,
                               learning_rate=0.9, min_step=1e-6,
                               deadline=deadline)
        
        # Powell refinement
        if time.time() < deadline - 10:
            points = _powell_refine(points, deadline=deadline, max_iters=40)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If no valid points found, return a basic configuration
    if best_points is None:
        best_points = _regular_polygon(n)
    
    return best_points


# EVOLVE-BLOCK-END
