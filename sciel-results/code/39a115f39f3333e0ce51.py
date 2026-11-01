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
    Objective function: Soft max of facet radii.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    # Soft max: (1/beta) * log(sum(exp(beta * r)))
    # To improve stability, shift by max
    r_max = np.max(radii)
    if r_max == 0:
        return 0.0
        
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    
    return log_sum_exp

def _spectral_energy(params):
    """
    Smooth spectral energy: sum_{i<j} (p_i · p_j)^2
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    G = X @ X.T
    off_diag_sq = np.sum(G**2) - N
    return off_diag_sq

def _spectral_gradient(params):
    """
    Gradient of spectral energy.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    G = X @ X.T
    grad = 4.0 * G @ X
    for i in range(N):
        grad[i] -= 4.0 * G[i, i] * X[i]
        
    dot_gx = np.sum(grad * X, axis=1, keepdims=True)
    grad_tan = grad - dot_gx * X
    
    return grad_tan.flatten()

def _spectral_relax(points, alpha_init=0.01, decay=0.999, max_iter=2000, deadline=None):
    """
    Spectral energy relaxation.
    """
    points = points.copy()
    alpha = alpha_init
    best_energy = _spectral_energy(points.flatten())
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        grad = _spectral_gradient(points.flatten()).reshape(N, 3)
        
        new_points = points - alpha * grad
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, np.newaxis]
        
        new_energy = _spectral_energy(new_points.flatten())
        
        if new_energy < best_energy:
            points = new_points
            best_energy = new_energy
            alpha *= 1.01
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
        
        alpha *= decay
        
    return points

def _lloyd_relax(points, steps=100, alpha=0.5, deadline=None):
    """
    Lloyd's algorithm for sphere covering.
    Moves each point towards the centroid of its Voronoi cell (approximated by incident facet circumcenters).
    """
    points = points.copy()
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        facets, circumcenters, radii = _compute_hull_data(points)
        if facets is None:
            break
            
        # Map point index to list of incident facet indices
        # facets is (F, 3) array of point indices
        # We need to know for each point i, which facets contain it.
        # Since N=50, we can just loop or use bincount logic.
        
        # Create a matrix M of shape (N, F) where M[i, f] = 1 if i is in facet f
        # But facets are triples.
        # Let's just collect circumcenters for each point.
        
        new_points = points.copy()
        
        # Efficient way: 
        # For each facet f, it has points p0, p1, p2 and circumcenter cc_f.
        # We want to sum cc_f for all f incident to point i.
        
        # Initialize sums
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for f in range(len(facets)):
            cc = circumcenters[f]
            # If circumcenter is zero (degenerate), skip or handle
            if np.linalg.norm(cc) < 1e-9:
                continue
            i0, i1, i2 = facets[f]
            sums[i0] += cc
            counts[i0] += 1
            sums[i1] += cc
            counts[i1] += 1
            sums[i2] += cc
            counts[i2] += 1
        
        # Compute centroids
        valid = counts > 0
        centroids = points[valid].copy()
        centroids[valid] = sums[valid] / counts[valid, np.newaxis]
        
        # Normalize centroids to unit sphere
        norms = np.linalg.norm(centroids, axis=1)
        norms[norms < 1e-9] = 1.0
        centroids = centroids / norms[:, np.newaxis]
        
        # Update points
        # new_points[valid] = (1-alpha)*points[valid] + alpha*centroids
        # Then renormalize
        update = points[valid] * (1 - alpha) + centroids * alpha
        update_norms = np.linalg.norm(update, axis=1)
        update_norms[update_norms < 1e-9] = 1.0
        update = update / update_norms[:, np.newaxis]
        
        new_points[valid] = update
        
        points = new_points
        
    return points

def _local_optimize(config, deadline):
    """
    Local optimization:
    1. Lloyd's relaxation
    2. L-BFGS on soft-max objective
    3. Lloyd's relaxation again
    4. Final L-BFGS polish
    """
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Lloyd's Relaxation
    if time.time() < deadline - 10.0:
        lloyd_config = _lloyd_relax(best_config, steps=50, alpha=0.5, deadline=deadline - 5.0)
        lloyd_score = _covering_radius_deg_hull(lloyd_config)
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_config = lloyd_config.copy()
            
    # Phase 2: L-BFGS with increasing beta
    betas = [10.0, 50.0, 200.0]
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
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
        
    # Phase 3: Final Lloyd Polish
    if time.time() < deadline - 2.0:
        lloyd_config = _lloyd_relax(best_config, steps=20, alpha=0.2, deadline=deadline - 1.0)
        lloyd_score = _covering_radius_deg_hull(lloyd_config)
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_config = lloyd_config.copy()
            
    # Phase 4: Very high beta L-BFGS
    if time.time() < deadline - 0.5:
        try:
            res = minimize(_objective, best_config.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50},
                           args=(500.0,))
            
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
    config = _spectral_relax(config, alpha_init=0.01, decay=0.999, max_iter=500, deadline=deadline - 60.0)
    
    # 3. Lloyd's Relaxation to get into a good basin
    config = _lloyd_relax(config, steps=50, alpha=0.5, deadline=deadline - 40.0)
    
    # 4. Local Optimization
    config, score = _local_optimize(config, deadline - 20.0)
    
    # 5. Multi-start: Try random perturbations and re-optimize
    # We have ~20s left.
    # Let's do 2-3 restarts.
    if time.time() < deadline - 10.0:
        for i in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb current best
            noise = np.random.randn(N, 3) * 0.2
            new_config = config + noise
            new_config = np.array([_normalize(p) for p in new_config])
            
            # Quick Lloyd relax
            new_config = _lloyd_relax(new_config, steps=20, alpha=0.5, deadline=deadline - 10.0)
            
            # Optimize
            new_config, new_score = _local_optimize(new_config, deadline - 5.0)
            
            if new_score < score:
                score = new_score
                config = new_config
                
    # Final check and normalization
    config = np.array([_normalize(p) for p in config])
    
    return config
