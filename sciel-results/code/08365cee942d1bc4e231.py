import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    Returns the maximum angular distance in radians.
    """
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(d, -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to ensure points are on sphere
        norms = np.linalg.norm(pts, axis=1)
        # If any norm is zero, return large value
        if np.any(norms < 1e-9):
            return 1e6
        pts = pts / norms[:, np.newaxis]
        # Compute covering radius
        return _covering_radius(pts)
    
    # Use L-BFGS-B to minimize the covering radius
    # This is a different approach from repulsion forces
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
    )
    
    # Extract optimized points
    opt_points = result.x.reshape(N, 3)
    
    # Normalize again to be safe
    norms = np.linalg.norm(opt_points, axis=1)
    opt_points = opt_points / norms[:, np.newaxis]
    
    # Compare with initial points to ensure improvement
    initial_radius = _covering_radius(points)
    opt_radius = _covering_radius(opt_points)
    
    if opt_radius < initial_radius:
        return opt_points
    else:
        return points
