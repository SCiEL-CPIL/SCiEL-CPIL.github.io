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
        # Vectorized triangle area computation
        n = len(points)
        idx = _triangle_indices(n)
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        # area = 0.5 * |cross(p2-p1, p3-p1)|
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = np.abs(cross) / 2.0
        a_min = np.min(areas)
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_objective(params, n=13, beta=50.0, idx=None):
    """
    Negative soft-min of triangle areas normalized by hull area.
    params: 26 values (13 x, 13 y)
    Uses log-sum-exp trick for smooth approximation to min.
    """
    if idx is None:
        idx = _triangle_indices(n)
    
    pts = params.reshape(n, 2)
    
    # Check validity
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 1e10
    
    if a_hull <= 1e-12:
        return 1e10
    
    # Triangle areas
    p1 = pts[idx[:, 0]]
    p2 = pts[idx[:, 1]]
    p3 = pts[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) / 2.0
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta approximates min(areas)
    # Use log-sum-exp for numerical stability
    min_area = np.min(areas)
    shifted = -beta * (areas - min_area)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = min_area + np.log(np.sum(np.exp(shifted))) / beta
    # Wait, that's soft max. For soft min:
    # soft_min(x) = -soft_max(-x) = -(1/beta)*log(sum(exp(-beta*x)))
    # = -min(x) - (1/beta)*log(sum(exp(-beta*(x - min(x)))))
    # Actually: soft_min(x) ≈ min(x) - (1/beta)*log(sum(exp(-beta*(x-min(x)))))
    # Hmm, let me just use: soft_min = -logsumexp(-beta * areas) / beta
    # = -logsumexp(-beta * areas) / beta
    
    # Recompute properly:
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    ratio = soft_min / a_hull
    return -ratio

def _objective_wrapper(params, beta, idx):
    return _soft_min_objective(params, 13, beta, idx)

def heilbronn_convex13() -> np.ndarray:
    """
    Maximize A_min / A_hull for 13 points using smooth optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    n = 13
    idx = _triangle_indices(n)
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_ratio = _compute_ratio(init_pts)
    best_points = init_pts.copy()
    
    # Try multiple beta values (higher beta = sharper min approximation)
    # and multiple random initializations
    rng = np.random.default_rng(42)
    
    for beta in [20.0, 40.0, 80.0, 150.0]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(4):
            if time.time() > deadline - 30:
                break
            
            # Create initial perturbation
            if trial == 0:
                x0 = init_pts.flatten().copy()
            else:
                # Random perturbation around regular polygon
                noise = rng.normal(0, 0.1, (n, 2))
                x0 = (init_pts + noise).flatten()
            
            # Optimize
            try:
                result = minimize(
                    _objective_wrapper, x0,
                    args=(beta, idx),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                
                pts = result.x.reshape(n, 2)
                ratio = _compute_ratio(pts)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Additional local refinement with Nelder-Mead on best found
    if time.time() < deadline - 20:
        try:
            def _nm_objective(p):
                pts = p.reshape(n, 2)
                return -_compute_ratio(pts)
            
            result = minimize(
                _nm_objective, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
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
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = init_pts
    
    return best_points

# EVOLVE-BLOCK-END
