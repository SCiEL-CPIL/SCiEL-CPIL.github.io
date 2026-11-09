# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import differential_evolution

def _compute_areas(points: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray) -> np.ndarray:
    """Compute areas for all triangles defined by indices."""
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return np.abs(cross) * 0.5

def _score(points: np.ndarray) -> float:
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
    
    if A_hull <= 1e-12:
        return 0.0
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    areas = _compute_areas(points, i_sel, j_sel, k_sel)
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull

def _objective(x: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray) -> float:
    """
    Objective for differential evolution: minimize -score.
    Uses exact min (not soft) since DE doesn't need gradients.
    """
    points = x.reshape(-1, 2)
    
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 1e-12:
        return 1e10
    
    areas = _compute_areas(points, i_sel, j_sel, k_sel)
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 1e10
    
    return -(min_area / A_hull)

def _de_optimize(x0: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray, time_limit: float, seed: int = 42) -> np.ndarray:
    """
    Run differential evolution with time limit.
    """
    n = x0.shape[0]
    bounds = [(0.01, 0.99)] * (2 * n)
    
    start_time = time.time()
    
    def callback(xk, convergence):
        if time.time() > time_limit:
            raise TimeoutError()
    
    try:
        result = differential_evolution(
            lambda x: _objective(x, i_sel, j_sel, k_sel),
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=seed,
            callback=callback,
            polish=True
        )
        return result.x.reshape(n, 2)
    except (TimeoutError, Exception):
        # If DE times out or fails, return x0
        return x0

def _nelder_mead_optimize(x0: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Run Nelder-Mead optimization as a local polish step.
    """
    from scipy.optimize import minimize
    n = x0.shape[0]
    
    start_time = time.time()
    
    def objective(x):
        if time.time() > time_limit:
            return 1e10
        return _objective(x, i_sel, j_sel, k_sel)
    
    try:
        result = minimize(
            objective,
            x0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        return result.x.reshape(n, 2)
    except Exception:
        return x0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area ratio.
    Uses differential evolution for global search + Nelder-Mead for local polish.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Precompute triangle indices
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    best_points = None
    best_score = -1.0
    rng = np.random.default_rng(42)
    
    # Initial configurations
    # 1. Grid-like (4x4 minus one corner)
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.05, 0.95)
    
    # 2. Hexagonal packing
    hex_pts = np.array([
        [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
        [0.35, 0.45], [0.65, 0.45],
        [0.2, 0.7], [0.5, 0.7], [0.8, 0.7],
        [0.35, 0.9], [0.65, 0.9],
        [0.5, 0.35], [0.35, 0.6], [0.65, 0.6]
    ])
    
    # 3. Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius = 0.4
    poly_pts = np.column_stack([0.5 + radius*np.cos(angles), 0.5 + radius*np.sin(angles)])
    poly_pts = np.vstack([poly_pts, [[0.5, 0.5]]])
    
    # 4. Known Heilbronn-like configuration: 3x3 grid + 4 extra points
    base_grid = np.array([
        [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
        [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
        [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]
    ])
    extra_pts = np.array([
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]
    ])
    heilbronn_like = np.vstack([base_grid, extra_pts])
    
    starts = [grid, hex_pts, poly_pts, heilbronn_like]
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 1: Differential evolution from best start (global search)
    if best_points is not None:
        de_time = min(30, (deadline - time.time()) * 0.4)
        if de_time > 5:
            improved = _de_optimize(best_points, i_sel, j_sel, k_sel, time_limit=time.time() + de_time, seed=42)
            improved_score = _score(improved)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved.copy()
    
    # Phase 2: Nelder-Mead polish on best point
    if best_points is not None:
        nm_time = min(20, (deadline - time.time()) * 0.3)
        if nm_time > 3:
            improved = _nelder_mead_optimize(best_points, i_sel, j_sel, k_sel, time_limit=time.time() + nm_time)
            improved_score = _score(improved)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved.copy()
    
    # Phase 3: Multiple DE restarts with different seeds
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        # Generate diverse starting points
        pts = rng.random((13, 2)) * 0.85 + 0.075
        
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Allocate time for DE on this restart
        remaining = deadline - 10 - time.time()
        if remaining < 10:
            break
        
        de_time = min(remaining * 0.5, 25)
        if de_time < 5:
            break
            
        improved = _de_optimize(pts, i_sel, j_sel, k_sel, time_limit=time.time() + de_time, seed=100 + restart_count)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
            
        # Quick NM polish if we have time
        if time.time() < deadline - 5:
            nm_time = min(8, deadline - 5 - time.time())
            if nm_time > 2:
                improved = _nelder_mead_optimize(best_points, i_sel, j_sel, k_sel, time_limit=time.time() + nm_time)
                improved_score = _score(improved)
                if improved_score > best_score:
                    best_score = improved_score
                    best_points = improved.copy()
    
    if best_points is None:
        best_points = rng.random((13, 2)) * 0.85 + 0.075
        
    return best_points

# EVOLVE-BLOCK-END
