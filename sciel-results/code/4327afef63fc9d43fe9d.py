"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        for simplex in facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            
            if n_pts == 3:
                a, b, c = pts[0], pts[1], pts[2]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                triple = np.dot(a, np.cross(b, c))
                denom = 4.0 * abs(triple)
                if denom < 1e-15:
                    continue
                sin_R = (np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)) / denom
                sin_R = min(sin_R, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, R)
            elif n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf


def _optimize_from_start(points, max_time=100.0):
    """Local optimization using Powell's method on spherical coordinates."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Convert to spherical coordinates (theta, phi)
    # theta: polar angle [0, pi], phi: azimuthal angle [0, 2*pi]
    thetas = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phis = np.arctan2(points[:, 1], points[:, 0])
    # Handle phi wrap-around issues? Powell doesn't like discontinuities.
    # It's better to optimize the 3D coordinates directly with a penalty or projection.
    # However, Powell on 2N angles is standard.
    
    def objective(x):
        # x is a flat vector of [theta_1, phi_1, theta_2, phi_2, ...]
        if time.time() - start_time > max_time:
            return best_radius # Stop condition handled by caller or timeout
        thetas = x[0::2]
        phis = x[1::2]
        
        # Convert back to cartesian
        pts = np.zeros((N, 3))
        pts[:, 2] = np.cos(thetas)
        pts[:, 0] = np.sin(thetas) * np.cos(phis)
        pts[:, 1] = np.sin(thetas) * np.sin(phis)
        
        # Normalize to be safe
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        return _covering_radius(pts)
    
    x0 = np.zeros(2 * N)
    x0[0::2] = thetas
    x0[1::2] = phis
    
    # Use Powell's method
    # maxiter is a limit on iterations, but we also check time
    result = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    if result.success or result.fun < best_radius:
        thetas = result.x[0::2]
        phis = result.x[1::2]
        new_points = np.zeros((N, 3))
        new_points[:, 2] = np.cos(thetas)
        new_points[:, 0] = np.sin(thetas) * np.cos(phis)
        new_points[:, 1] = np.sin(thetas) * np.sin(phis)
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        new_points = new_points / norms
        
        new_radius = _covering_radius(new_points)
        if new_radius < best_radius:
            best_points = new_points
            best_radius = new_radius
            
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius(best_points)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random start
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize from this start
        opt_points, opt_radius = _optimize_from_start(rand_points, max_time=20.0)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Final optimization from the best point found
    if time.time() < deadline:
        final_points, final_radius = _optimize_from_start(best_points, max_time=30.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
