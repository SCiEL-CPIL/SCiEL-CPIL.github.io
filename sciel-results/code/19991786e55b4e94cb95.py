import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import scipy.linalg

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

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
    Objective function: Sum of exp(beta * r_i^2).
    This emphasizes large radii while keeping the function smooth.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    # Use squared radii in radians to make the penalty sharper for larger angles
    # r is in radians
    # We want to minimize max(r). 
    # Sum(exp(beta * r^2)) is a good surrogate.
    # If beta is large, it approximates max(r^2) * N.
    # We can also just use sum(r**power) with high power.
    # Let's stick to log-sum-exp style for max:
    # max(r) ~ (1/beta) * log(sum(exp(beta * r)))
    # But r is in radians.
    
    # To be safe and smooth, let's use sum(r**4) or similar.
    # Actually, the gradient of max is problematic. 
    # Let's use a soft max:
    r_max = np.max(radii)
    if r_max == 0:
        return 0.0
    
    # Shift for stability
    shifted = beta * radii
    shifted -= np.max(shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + np.max(radii)
    
    return log_sum_exp

def _spectral_energy(params):
    """
    Smooth spectral energy: -trace(Gram matrix) = -sum_{i<j} (p_i · p_j)^2
    Minimizing this promotes uniform distribution on sphere.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    # Gram matrix G = X @ X.T, diagonal is 1
    G = X @ X.T
    # trace(G) = N (diagonal) + sum_{i≠j} G_ij
    # We want to maximize trace(G) which means maximizing sum of pairwise dot products
    # But that's not right - we want uniform spread, so we want MINIMIZE sum of squared dot products
    # Actually for uniform distribution, we want the points to be as orthogonal as possible
    # So minimize sum_{i≠j} (p_i · p_j)^2
    off_diag_sq = np.sum(G**2) - N  # subtract diagonal
    return off_diag_sq

def _spectral_gradient(params):
    """
    Gradient of spectral energy with respect to parameters.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    G = X @ X.T
    # Gradient of sum_{i≠j} G_ij^2 w.r.t. X
    # d/dX_k [sum_{i≠j} (X_i · X_j)^2] = 4 * sum_{j≠k} (X_k · X_j) X_j
    grad = 4.0 * G @ X
    # Zero out diagonal contribution
    for i in range(N):
        grad[i] -= 4.0 * G[i, i] * X[i]
    
    # Project gradient to tangent space
    dot_gx = np.sum(grad * X, axis=1, keepdims=True)
    grad_tan = grad - dot_gx * X
    
    return grad_tan.flatten()

def _repulsion_relax(points, alpha_init=0.1, decay=0.999, max_iter=1000, deadline=None):
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
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        alpha *= decay
        
    return points

def _spectral_relax(points, alpha_init=0.01, decay=0.999, max_iter=2000, deadline=None):
    """
    Spectral energy relaxation: minimize sum_{i≠j} (p_i · p_j)^2
    using gradient descent with adaptive step size.
    """
    points = points.copy()
    alpha = alpha_init
    best_energy = _spectral_energy(points.flatten())
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        grad = _spectral_gradient(points.flatten()).reshape(N, 3)
        
        # Adaptive step size: reduce if energy increases
        new_points = points - alpha * grad
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, np.newaxis]
        
        new_energy = _spectral_energy(new_points.flatten())
        
        if new_energy < best_energy:
            points = new_points
            best_energy = new_energy
            alpha *= 1.01  # slightly increase step
        else:
            alpha *= 0.9  # reduce step
            if alpha < 1e-6:
                break
        
        alpha *= decay
        
    return points

def _local_optimize(config, deadline):
    """
    Local optimization using L-BFGS on the soft-max objective,
    followed by spectral energy polishing.
    """
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Soft-max optimization with varying beta
    betas = [5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 150},
                           args=(beta,))
            
            new_config = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_config, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_config = new_config / norms
            
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_score:
                best_score = new_score
                best_config = new_config.copy()
                
        except Exception:
            pass
    
    # Phase 2: Spectral energy polishing
    if time.time() < deadline - 5.0:
        try:
            spectral_config = _spectral_relax(best_config, alpha_init=0.005, decay=0.999, 
                                               max_iter=500, deadline=deadline - 2.0)
            spectral_score = _covering_radius_deg_hull(spectral_config)
            if spectral_score < best_score:
                best_score = spectral_score
                best_config = spectral_config.copy()
        except Exception:
            pass
    
    # Phase 3: Final fine-tuning with very high beta
    if time.time() < deadline - 1.0:
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
                           args=(100.0,))
            
            new_config = res.x.reshape(N, 3)
            norms = np.linalg.norm(new_config, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_config = new_config / norms
            
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_score:
                best_score = new_score
                best_config = new_config.copy()
                
        except Exception:
            pass
            
    return best_config, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # 1. Generate initial configuration: Fibonacci spiral
    config = _fib_sample(N)
    
    # 2. Spectral energy relaxation first (smooth, fast)
    config = _spectral_relax(config, alpha_init=0.01, decay=0.999, max_iter=1000, deadline=deadline - 80.0)
    
    # 3. Repulsion Relaxation to spread points evenly
    config = _repulsion_relax(config, alpha_init=0.05, decay=0.999, max_iter=300, deadline=deadline - 50.0)
    
    # 4. Local Optimization
    config, score = _local_optimize(config, deadline - 20.0)
    
    # 5. Multi-start: Try a few random perturbations and re-optimize
    
    if time.time() < deadline - 30.0:
        for i in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb current best
            noise = np.random.randn(N, 3) * 0.1
            new_config = config + noise
            new_config = np.array([_normalize(p) for p in new_config])
            
            # Quick spectral + repulsion relax
            new_config = _spectral_relax(new_config, alpha_init=0.005, decay=0.999, 
                                         max_iter=200, deadline=deadline - 20.0)
            new_config = _repulsion_relax(new_config, alpha_init=0.02, decay=0.999, 
                                          max_iter=100, deadline=deadline - 15.0)
            
            # Optimize
            new_config, new_score = _local_optimize(new_config, deadline - 10.0)
            
            if new_score < score:
                score = new_score
                config = new_config
                
    # Final check and normalization
    config = np.array([_normalize(p) for p in config])
    
    return config
