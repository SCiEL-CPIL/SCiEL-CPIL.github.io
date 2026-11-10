# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_indices(n=13):
    """Precompute all triangle index triples for n points."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

def _compute_ratio(points):
    """Compute A_min / A_hull ratio."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 1e-12:
            return 0.0
        n = len(points)
        idx = _triangle_indices(n)
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = np.abs(cross) / 2.0
        a_min = np.min(areas)
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_objective(params, n=13, beta=50.0, idx=None):
    """
    Negative soft-min of triangle areas normalized by hull area.
    Uses log-sum-exp trick for smooth approximation to min.
    """
    if idx is None:
        idx = _triangle_indices(n)
    
    pts = params.reshape(n, 2)
    
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 1e10
    
    if a_hull <= 1e-12:
        return 1e10
    
    p1 = pts[idx[:, 0]]
    p2 = pts[idx[:, 1]]
    p3 = pts[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) / 2.0
    
    # Soft min: -logsumexp(-beta * areas) / beta
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    ratio = soft_min / a_hull
    return -ratio

def _objective_wrapper(params, beta, idx):
    return _soft_min_objective(params, 13, beta, idx)

def _grid_initialization(grid_size, rng, n=13):
    """
    Generate diverse initial point sets from a grid.
    """
    grid_vals = np.linspace(-1.5, 1.5, grid_size)
    points = np.zeros((n, 2))
    for i in range(n):
        gx = rng.integers(0, grid_size)
        gy = rng.integers(0, grid_size)
        jitter = rng.normal(0, 0.02, 2)
        points[i] = [grid_vals[gx] + jitter[0], grid_vals[gy] + jitter[1]]
    return points

def _refine_with_gradient(pts, idx, deadline, max_iter=150):
    """
    Refine points using L-BFGS-B with increasing beta for sharper min approximation.
    """
    best_ratio = _compute_ratio(pts)
    best_points = pts.copy()
    
    betas = [30.0, 60.0, 120.0]
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective_wrapper, x0,
                args=(beta, idx),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            refined_pts = result.x.reshape(13, 2)
            ratio = _compute_ratio(refined_pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_pts.copy()
        except Exception:
            pass
    
    return best_points, best_ratio

def heilbronn_convex13() -> np.ndarray:
    """
    Maximize A_min / A_hull for 13 points using discrete-continuous hybrid search.
    Strategy:
    1. Generate diverse initial configurations (regular polygon + grid + random)
    2. Refine each with gradient-based optimization (annealed beta schedule)
    3. Track the best configuration found
    4. Final high-precision polish
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit
    
    n = 13
    idx = _triangle_indices(n)
    rng = np.random.default_rng(12345)
    
    # Start with regular 13-gon as baseline
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_ratio = _compute_ratio(init_pts)
    best_points = init_pts.copy()
    
    # Phase 1: Multiple diverse initializations + gradient refinement
    # Try regular polygon with different perturbations
    for trial in range(4):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            x0_pts = init_pts.copy()
        else:
            noise = rng.normal(0, 0.1 * (trial + 1), (n, 2))
            x0_pts = init_pts + noise
        
        # Refine with gradient optimization
        refined_pts, ratio = _refine_with_gradient(x0_pts, idx, deadline, max_iter=120)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_pts.copy()
    
    # Phase 2: Grid-based diverse initialization
    grid_configs = [
        (8, 3),    # 8x8 grid, 3 trials
        (10, 2),   # 10x10 grid, 2 trials
        (12, 2),   # 12x12 grid, 2 trials
    ]
    
    for grid_size, n_trials in grid_configs:
        if time.time() > deadline - 45:
            break
        
        for trial in range(n_trials):
            if time.time() > deadline - 30:
                break
            
            grid_pts = _grid_initialization(grid_size, rng, n)
            
            try:
                hull = ConvexHull(grid_pts)
                if hull.volume < 0.1:
                    continue
            except Exception:
                continue
            
            refined_pts, ratio = _refine_with_gradient(grid_pts, idx, deadline, max_iter=100)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_pts.copy()
    
    # Phase 3: Random perturbations around best solution
    if time.time() < deadline - 30:
        for trial in range(6):
            if time.time() > deadline - 15:
                break
            
            noise = rng.normal(0, 0.05, (n, 2))
            perturbed = best_points + noise
            
            refined_pts, ratio = _refine_with_gradient(perturbed, idx, deadline, max_iter=80)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_pts.copy()
    
    # Phase 4: Final high-precision polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            def _nm_objective(p):
                pts = p.reshape(n, 2)
                return -_compute_ratio(pts)
            
            result = minimize(
                _nm_objective, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            pts = result.x.reshape(n, 2)
            ratio = _compute_ratio(pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 5: One more L-BFGS-B polish with very high beta for final sharpness
    if time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            result = minimize(
                _objective_wrapper, x0,
                args=(300.0, idx),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            pts = result.x.reshape(n, 2)
            ratio = _compute_ratio(pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = pts.copy()
        except Exception:
            pass
    
    # Final validation
    if best_points is None or best_points.shape != (13, 2):
        best_points = init_pts
    
    if not np.all(np.isfinite(best_points)):
        best_points = init_pts
    
    return best_points

# EVOLVE-BLOCK-END
