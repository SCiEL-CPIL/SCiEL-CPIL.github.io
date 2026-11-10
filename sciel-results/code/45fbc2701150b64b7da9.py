import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    Fast vectorized implementation.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback for degenerate cases

    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
        
    # Distance from origin to the plane of the facet
    # d = |normal . p0| / |normal|
    # This is cos(alpha) where alpha is the angular circumradius
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    
    # Clip to valid range for arccos
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    return np.max(alphas)

def _covering_radius_smooth(points, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
        
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    
    # Log-sum-exp for smooth max
    max_alpha = np.max(alphas)
    shifted = alphas - max_alpha
    log_sum_exp = max_alpha + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _repulsive_relaxation(points, steps=500, alpha=0.1, time_limit=None, start_time=None):
    """
    Relax points using repulsive forces to spread them out.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Compute all pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        points += alpha * forces
        points = _normalize(points)
        
    return points

def _covering_force(points):
    """
    Compute forces that push points toward regions of high covering radius.
    Uses the gradient of the smooth covering radius.
    """
    n = points.shape[0]
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: uniform repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        return forces
    
    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return np.zeros_like(points)
        
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    
    # For each point, compute the force based on adjacent facets
    # The force should push points toward larger facets (higher circumradius)
    forces = np.zeros_like(points)
    
    # For each simplex, add force to its vertices proportional to the circumradius
    for i, simplex in enumerate(simplices):
        if not mask[i]:
            continue
        weight = alphas[i]
        for vertex_idx in simplex:
            # Force direction: away from the facet normal (toward the center of the facet's cap)
            normal_dir = normals[i] / norms[i]
            # The force on each vertex should push it toward the circumcenter of its Voronoi cell
            # Approximation: push away from other points, weighted by local covering radius
            forces[vertex_idx] += weight * normal_dir
    
    return forces

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def time_remaining():
        return deadline - time.time()

    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)

    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Relax the initial configuration
    relaxed = _repulsive_relaxation(points, steps=1000, alpha=0.05, time_limit=5, start_time=start_time)
    rad = _covering_radius(relaxed)
    if rad < best_radius:
        best_points = relaxed
        best_radius = rad

    # 2. Define Objective for Optimizer
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)

    # 3. Local Optimization using BFGS with smooth surrogate, then Nelder-Mead polish
    
    def objective_smooth(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius_smooth(pts, beta=50.0)

    def optimize_from(points):
        nonlocal best_points, best_radius
        params0 = points.flatten()
        
        # Phase 1: BFGS with smooth surrogate
        try:
            result_bfgs = minimize(objective_smooth, params0, method='BFGS',
                                   options={'maxiter': 500, 'gtol': 1e-8, 'disp': False})
            if result_bfgs.fun < best_radius:
                best_points = _normalize(result_bfgs.x.reshape(N, 3))
                best_radius = _covering_radius(best_points)
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead polish with exact objective
        try:
            result_nm = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result_nm.fun < best_radius:
                best_points = _normalize(result_nm.x.reshape(N, 3))
                best_radius = result_nm.fun
        except Exception:
            pass

    # Run optimization from the best relaxed point
    optimize_from(best_points)
    
    # 4. Random Restarts with gradient-based relaxation
    while time_remaining() > 10:
        # Perturb current best
        noise_scale = 0.3
        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        perturbed = best_points + noise
        perturbed = _normalize(perturbed)
        
        # Quick relaxation to get out of the noise
        relaxed = _repulsive_relaxation(perturbed, steps=200, alpha=0.05, 
                                        time_limit=3, start_time=start_time)
        rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            
            # Optimize from this new start
            optimize_from(best_points)
            
            # If we found a significantly better solution, try a few more perturbations from it
            # to see if we can improve further immediately
            if time_remaining() > 20:
                for _ in range(2):
                    noise = np.random.normal(0, 0.1, size=best_points.shape)
                    perturbed = best_points + noise
                    perturbed = _normalize(perturbed)
                    relaxed = _repulsive_relaxation(perturbed, steps=100, alpha=0.05, 
                                                    time_limit=2, start_time=start_time)
                    rad = _covering_radius(relaxed)
                    if rad < best_radius:
                        best_points = relaxed
                        best_radius = rad
                        optimize_from(best_points)

    # 5. Final Polish with tighter tolerance
    if time_remaining() > 5:
        def objective_tight(params):
            if time.time() > deadline:
                return 1e6
            pts = params.reshape(N, 3)
            pts = _normalize(pts)
            if not np.all(np.isfinite(pts)):
                return 1e6
            return _covering_radius(pts)
        
        try:
            result_final = minimize(objective_tight, best_points.flatten(), method='Nelder-Mead',
                                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result_final.fun < best_radius:
                best_points = _normalize(result_final.x.reshape(N, 3))
                best_radius = result_final.fun
        except Exception:
            pass

    return best_points
