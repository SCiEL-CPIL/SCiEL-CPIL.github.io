# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute A_min / A_hull for a set of points.
    Vectorized for speed.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = scipy.spatial.ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-12:
        return 0.0
        
    # Compute all triangle areas vectorized
    # Indices for all C(n, 3) triangles
    # Precomputed globally for speed if possible, but n is fixed at 13.
    # We can generate them once outside if we want, but inside is fine for 13 points.
    # Let's use itertools for clarity, or precompute indices.
    # Precomputing indices is faster.
    
    # Since n=13 is fixed, we can hardcode or cache indices.
    # But for flexibility, we compute here.
    idx = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # v1 = p2 - p1, v2 = p3 - p1
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product z-component: v1_x * v2_y - v1_y * v2_x
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / A_hull

def _local_search(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Perform local search (hill climbing) to improve the point configuration.
    Uses 8-direction moves with adaptive step size.
    """
    best_points = points.copy()
    best_score = _compute_score(best_points)
    n = len(best_points)
    
    # Initial step size
    step = 0.1
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 10
    
    # Directions: 8 neighbors
    directions = [
        [0, 1], [0, -1], [1, 0], [-1, 0],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ]
    # Normalize diagonal directions? No, just use raw steps.
    # Actually, better to use normalized directions to have consistent step magnitude.
    # Let's stick to axis-aligned and diagonal with same step size for simplicity, 
    # or just axis-aligned for fewer evaluations? 
    # 8 directions * 13 points = 104 evaluations per sweep.
    # If each eval is ~1ms, 104ms per sweep. 
    # 10 sweeps = 1s. 
    # This is fast enough.
    
    while time.time() < deadline - 10:
        improved = False
        for i in range(n):
            for dx, dy in directions:
                candidate = best_points.copy()
                candidate[i, 0] += dx * step
                candidate[i, 1] += dy * step
                
                # Optional: Keep points in a reasonable range to avoid numerical issues
                # The metric is scale invariant, but large numbers can cause precision loss.
                # We can normalize points occasionally or just let it be.
                # For now, just check finiteness.
                if not np.all(np.isfinite(candidate)):
                    continue
                
                score = _compute_score(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        if not improved:
            step *= 0.8
            if step < min_step:
                break
        else:
            # If improved, keep step size or slightly increase? 
            # Usually keep or slightly increase to explore.
            # Let's keep it stable to converge.
            pass
            
    return best_points

def _refine(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Use scipy.optimize.minimize to polish the solution.
    """
    n = len(points)
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        # Penalty for non-finite
        if not np.all(np.isfinite(pts)):
            return 1e6
        score = _compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score
    
    try:
        res = scipy.optimize.minimize(
            objective,
            points.flatten(),
            method='Powell',
            options={'maxiter': 200, 'ftol': 1e-10, 'xatol': 1e-10, 'disp': False}
        )
        candidate = res.x.reshape(n, 2)
        # Verify candidate is better
        if _compute_score(candidate) > _compute_score(points):
            return candidate
    except Exception:
        pass
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) of finite point coordinates.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Try some structured initializations
    initializations = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(init_poly)
    
    # Grid-like (4x4 grid, take 13 points)
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    init_grid = np.array(grid_pts[:n])
    initializations.append(init_grid)
    
    # 2. Run local search + refinement on structured starts
    for init in initializations:
        if time.time() > deadline - 60:
            break
            
        # Local search
        pts = _local_search(init, deadline)
        # Refine
        pts = _refine(pts, deadline)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # 3. Random restarts
    rng = np.random.default_rng(42)
    restart_count = 0
    
    while time.time() < deadline - 30:
        # Generate random points in a square
        # Scale doesn't matter, but let's use [-1, 1]
        init_rand = rng.uniform(-1, 1, (n, 2))
        
        # Local search
        pts = _local_search(init_rand, deadline)
        # Refine
        pts = _refine(pts, deadline)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
        restart_count += 1
        
        # If we've done many restarts without improvement, we might be stuck in local minima.
        # But random restarts help escape.
        
    if best_points is None:
        # Fallback
        best_points = init_poly
        
    return best_points

# EVOLVE-BLOCK-END
