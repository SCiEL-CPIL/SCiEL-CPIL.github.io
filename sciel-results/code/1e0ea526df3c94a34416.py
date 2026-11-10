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

    # 3. Local Optimization using Nelder-Mead on Cartesian coordinates
    # We optimize on the sphere by re-normalizing in the objective.
    # Nelder-Mead works on the parameter space.
    
    def optimize_from(points):
        nonlocal best_points, best_radius
        params0 = points.flatten()
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_radius:
                best_points = _normalize(result.x.reshape(N, 3))
                best_radius = result.fun
        except Exception:
            pass

    # Run optimization from the best relaxed point
    optimize_from(best_points)
    
    # 4. Random Restarts
    # Use time to try different starting points or perturbations
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

    # 5. Final Polish
    if time_remaining() > 2:
        optimize_from(best_points)

    return best_points
