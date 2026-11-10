import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback for degenerate cases

    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]

        # Normal to the plane of the triangle
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)

        if norm_normal < 1e-12:
            continue

        # Distance from origin to the plane
        d = np.abs(np.dot(normal, p0)) / norm_normal

        # The circumradius (angular) alpha satisfies cos(alpha) = d
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)

        if alpha > max_radius:
            max_radius = alpha

    return max_radius

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _repulsive_relaxation(points, steps=1000, alpha=0.1, time_limit=None, start_time=None, force_exponent=2.0):
    """
    Relax points using repulsive forces to spread them out.
    Vectorized force calculation for speed.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Compute all pairwise differences: shape (n, n, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = dist_sq > 1e-12
        # Force magnitude: 1 / dist^force_exponent for unit vectors (repulsive)
        inv_dist = np.zeros_like(dist_sq)
        inv_dist[mask] = 1.0 / (dist_sq[mask] ** (force_exponent / 2.0))
        
        forces = np.sum(diff * inv_dist[:, :, np.newaxis], axis=1)
        
        # Adaptive step size that decreases over time
        step_size = alpha * (1.0 - step / steps)
        
        # Update
        points += step_size * forces
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

    # 2. Repulsive Relaxation (Faster with vectorization)
    points = _repulsive_relaxation(points, steps=3000, alpha=0.08, time_limit=15, start_time=start_time, force_exponent=2.0)
    current_radius = _covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius

    # 3. Local Optimization using Spherical Coordinates and Nelder-Mead
    def spherical_to_cartesian(params):
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def cartesian_to_spherical(pts):
        theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        params = np.zeros(2 * N)
        params[0::2] = theta
        params[1::2] = phi
        return params

    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)

    # Initialize params from best points
    params0 = cartesian_to_spherical(best_points)

    try:
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_radius:
            best_points = spherical_to_cartesian(result.x)
            best_radius = result.fun
    except Exception:
        pass

    # 4. Random Restarts with Local Optimization
    restart_count = 0
    while time_remaining() > 20:
        restart_count += 1
        # Use different noise scales for diversity
        noise_scales = [0.1, 0.15, 0.2, 0.3, 0.5]
        noise_scale = noise_scales[restart_count % len(noise_scales)]
        
        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        perturbed = best_points + noise
        perturbed = _normalize(perturbed)
        
        # Quick relaxation with different force exponents
        force_exp = [2.0, 2.5, 3.0][restart_count % 3]
        relaxed = _repulsive_relaxation(perturbed, steps=800, alpha=0.06, 
                                        time_limit=8, start_time=start_time, force_exponent=force_exp)
        rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            
            # Optimize from this new start
            params0 = cartesian_to_spherical(best_points)
            try:
                result = minimize(objective, params0, method='Nelder-Mead',
                                  options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if result.fun < best_radius:
                    best_points = spherical_to_cartesian(result.x)
                    best_radius = result.fun
            except Exception:
                pass

    # 5. Final Polish if time permits
    if time_remaining() > 10:
        params0 = cartesian_to_spherical(best_points)
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_radius:
                best_points = spherical_to_cartesian(result.x)
                best_radius = result.fun
        except Exception:
            pass

    return best_points
