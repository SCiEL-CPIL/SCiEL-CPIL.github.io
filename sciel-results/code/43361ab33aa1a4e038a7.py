import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize a batch of vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii.
    Returns facets, circumcenters, radii.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        return 180.0 # Invalid or degenerate
    return np.degrees(np.max(radii))

def _objective(params, beta=10.0):
    """
    Objective function: Soft max of facet radii.
    Uses log-sum-exp for stability.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    if np.max(radii) == 0:
        return 0.0
        
    # Soft max approximation: max(r) ~ (1/beta) * log(sum(exp(beta * r)))
    shifted = beta * radii
    shifted -= np.max(shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + np.max(radii)
    
    return log_sum_exp

def _repulsion_relax(points, alpha_init=0.05, decay=0.999, max_iter=1000, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        points = points + alpha * forces_tan
        points = _normalize(points)
        
        alpha *= decay
        
    return points

def _local_optimize(config, deadline):
    """
    Local optimization using L-BFGS on the soft-max objective.
    """
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Try a few beta values to balance smoothness and fidelity to max
    betas = [5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200},
                           args=(beta,))
            
            new_config = res.x.reshape(N, 3)
            new_config = _normalize(new_config)
            
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_score:
                best_score = new_score
                best_config = new_config.copy()
                
        except Exception:
            pass
            
    return best_config, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Initialize with Fibonacci spiral
    config = _fib_sample(N)
    
    # 2. Repulsion Relaxation to spread points evenly
    # This helps get out of the "grid" structure of Fibonacci
    config = _repulsion_relax(config, alpha_init=0.05, decay=0.999, max_iter=500, deadline=deadline - 60.0)
    
    # 3. Local Optimization
    config, score = _local_optimize(config, deadline - 20.0)
    
    # 4. Multi-start: Try a few random perturbations and re-optimize
    # If we have time, try 3 more random starts
    
    if time.time() < deadline - 30.0:
        for i in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb current best
            noise = np.random.randn(N, 3) * 0.1
            new_config = config + noise
            new_config = _normalize(new_config)
            
            # Quick relax
            new_config = _repulsion_relax(new_config, alpha_init=0.02, decay=0.999, max_iter=100, deadline=deadline - 15.0)
            
            # Optimize
            new_config, new_score = _local_optimize(new_config, deadline - 10.0)
            
            if new_score < score:
                score = new_score
                config = new_config
                
    # Final check and normalization
    config = _normalize(config)
    
    return config
