# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

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

def _smooth_metric(points: np.ndarray, alpha=10.0) -> float:
    """
    Smoothed objective to maximize.
    Uses soft-min and soft-max to approximate d_min and d_max smoothly.
    We want to maximize (d_min / d_max)^2.
    Equivalent to minimizing -2 * (log(d_min) - log(d_max)).
    """
    dists = pdist(points)
    # Soft min: approximates min(dists)
    # log-sum-exp trick for stability
    m = np.min(dists)
    soft_min = m + np.log(np.mean(np.exp(dists - m)))
    
    # Soft max: approximates max(dists)
    M = np.max(dists)
    soft_max = M + np.log(np.mean(np.exp(dists - M)))
    
    if soft_min <= 0 or soft_max <= 0:
        return -1e10
        
    # We want to maximize (soft_min / soft_max)^2
    # So minimize the negative log of this ratio
    # log(ratio^2) = 2 * (log(soft_min) - log(soft_max))
    return -2.0 * (np.log(soft_min) - np.log(soft_max))

def _get_initial_points(seed: int) -> np.ndarray:
    """Generate diverse initial configurations."""
    np.random.seed(seed)
    n = 14
    d = 3
    
    # Strategy 1: Random Gaussian
    if seed % 3 == 0:
        return np.random.randn(n, d)
    
    # Strategy 2: Random on Sphere (normalized)
    if seed % 3 == 1:
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts
        
    # Strategy 3: Grid-like / Icosahedron based perturbation
    # Simple icosahedron vertices (12 points) + 2 perturbations
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 1, phi], [-1/phi, 1, phi], [1/phi, -1, phi], [-1/phi, -1, phi],
        [1, phi, 1/phi], [1, phi, -1/phi], [1, -phi, 1/phi], [1, -phi, -1/phi],
        [1/phi, phi, 1], [-1/phi, phi, 1], [1/phi, -phi, 1], [-1/phi, -phi, 1],
        [phi, 1, 1/phi], [phi, 1, -1/phi], [phi, -1, 1/phi], [phi, -1, -1/phi],
        [1, 1, phi], [1, 1, -phi], [1, -1, phi], [1, -1, -phi]
    ])
    # Normalize icosahedron
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Pick 12 random vertices
    idx = np.random.choice(len(ico), 12, replace=False)
    pts = ico[idx]
    # Add 2 random points near the gaps
    p1 = np.random.randn(3) * 0.5 + pts[0]
    p2 = np.random.randn(3) * 0.5 + pts[1]
    pts = np.vstack([pts, p1, p2])
    return pts

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_points, best_score
        try:
            pts = np.asarray(pts, dtype=float)
            if pts.shape != (n, d):
                return
            if not np.all(np.isfinite(pts)):
                return
            score = _evaluate(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass

    # Initial guess
    init_pts = _get_initial_points(42)
    update_best(init_pts)

    # Multi-start Nelder-Mead optimization
    # We optimize in 3D space directly.
    # To avoid scaling issues, we might normalize occasionally, but Nelder-Mead
    # works best if the objective is well-scaled.
    # The smoothed metric is scale-invariant in terms of ratio, but the gradient
    # magnitude depends on scale. We'll rely on the soft-min/max handling.
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        x0 = _get_initial_points(1000 + r)
        
        # Flatten points to 1D for scipy.optimize
        x0_flat = x0.flatten()
        
        def objective(x_flat):
            pts = x_flat.reshape(n, d)
            # Add a small penalty if points are too close to zero or inf
            if not np.all(np.isfinite(pts)):
                return 1e10
            return _smooth_metric(pts, alpha=10.0)
        
        try:
            result = minimize(
                objective, 
                x0_flat, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            best_pts_from_opt = result.x.reshape(n, d)
            update_best(best_pts_from_opt)
        except Exception:
            pass

    # Final polish with a tighter tolerance on the best found
    if best_points is not None and time.time() < deadline - 10:
        x0_flat = best_points.flatten()
        def objective_tight(x_flat):
            pts = x_flat.reshape(n, d)
            if not np.all(np.isfinite(pts)):
                return 1e10
            # Increase alpha for sharper min/max approximation
            return _smooth_metric(pts, alpha=50.0)
            
        try:
            result = minimize(
                objective_tight, 
                x0_flat, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            best_pts_final = result.x.reshape(n, d)
            update_best(best_pts_final)
        except Exception:
            pass

    if best_points is None:
        best_points = init_pts

    return best_points

# EVOLVE-BLOCK-END
