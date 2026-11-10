import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _facet_alphas(points):
    """Return per-facet angular circumradius (radians). Returns empty array if hull fails."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([])
    try:
        simplices = hull.simplices
    except Exception:
        return np.array([])
    if simplices.size == 0:
        return np.array([])
    tri = points[simplices]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    nrm = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(nrm, axis=1)
    mask = nn > 1e-12
    if not np.any(mask):
        return np.array([])
    d = np.abs(nrm[mask] @ p0[mask].T) / nn[mask]
    d = np.clip(d, 0.0, 1.0)
    return np.arccos(d)

def _covering_radius(points):
    """Exact covering radius in radians (max facet circumradius)."""
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return np.pi
    return float(np.max(alphas))

def _smooth_obj(params, beta=50.0):
    """
    Smooth surrogate: log-sum-exp of facet circumradii scaled by beta.
    As beta -> inf, this approaches the max.
    """
    if time.time() > _DEADLINE:
        return 1e6
    pts = params.reshape(N, 2)
    theta = pts[:, 0]
    phi = pts[:, 1]
    
    # Convert spherical to Cartesian
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    points = _normalize(points)
    
    if not np.all(np.isfinite(points)):
        return 1e6
    
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return 1e6
    
    # Log-sum-exp for numerical stability
    m = np.max(alphas)
    if m > 1.0:
        return 1e6
    val = m + np.log(np.sum(np.exp(alphas - m))) / beta
    return float(val)

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

def _cartesian_to_spherical(pts):
    """Convert unit sphere points to (theta, phi) parameters."""
    z = np.clip(pts[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    # Handle phi wrapping
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    params = np.column_stack([theta, phi])
    return params

def _spherical_to_cartesian(params):
    """Convert (theta, phi) parameters to unit sphere points."""
    theta = params[:, 0]
    phi = params[:, 1]
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    return _normalize(points)

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140.0  # Safety margin before 150s limit

    def time_remaining():
        return _DEADLINE - time.time()

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

    # 2. Define smooth objective for L-BFGS-B
    def smooth_objective(params):
        return _smooth_obj(params, beta=50.0)

    # 3. Local Optimization using L-BFGS-B on spherical coordinates
    def optimize_smooth(points, beta=50.0):
        """Optimize using L-BFGS-B with smooth surrogate."""
        params0 = _cartesian_to_spherical(points)
        
        # Bounds: theta in [0, pi], phi in [-pi, pi]
        bounds = [(0.0, np.pi)] * N + [(-np.pi, np.pi)] * N
        # Reshape bounds to (N, 2) format
        bounds_2d = [(0.0, np.pi), (-np.pi, np.pi)] * N
        
        try:
            result = minimize(smooth_objective, params0.flatten(), method='L-BFGS-B',
                              bounds=bounds_2d,
                              options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
            if result.fun < best_radius and result.success:
                new_points = _spherical_to_cartesian(result.x.reshape(N, 2))
                actual_radius = _covering_radius(new_points)
                if actual_radius < best_radius:
                    return new_points, actual_radius
        except Exception:
            pass
        
        return None, best_radius

    # Run optimization from the best relaxed point
    new_pts, new_rad = optimize_smooth(best_points, beta=50.0)
    if new_pts is not None:
        best_points = new_pts
        best_radius = new_rad

    # 4. Random Restarts with smooth optimization
    restart_count = 0
    while time_remaining() > 10:
        restart_count += 1
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
            new_pts, new_rad = optimize_smooth(best_points, beta=50.0)
            if new_pts is not None:
                best_points = new_pts
                best_radius = new_rad
            
            # If we found a significantly better solution, try a few more perturbations from it
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
                        new_pts, new_rad = optimize_smooth(best_points, beta=50.0)
                        if new_pts is not None:
                            best_points = new_pts
                            best_radius = new_rad

    # 5. Final Polish with tighter tolerance
    if time_remaining() > 2:
        def tight_objective(params):
            return _smooth_obj(params, beta=100.0)
        
        params0 = _cartesian_to_spherical(best_points)
        bounds_2d = [(0.0, np.pi), (-np.pi, np.pi)] * N
        
        try:
            result = minimize(tight_objective, params0.flatten(), method='L-BFGS-B',
                              bounds=bounds_2d,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000})
            if result.fun < best_radius and result.success:
                new_points = _spherical_to_cartesian(result.x.reshape(N, 2))
                actual_radius = _covering_radius(new_points)
                if actual_radius < best_radius:
                    best_points = new_points
                    best_radius = actual_radius
        except Exception:
            pass

    return best_points
