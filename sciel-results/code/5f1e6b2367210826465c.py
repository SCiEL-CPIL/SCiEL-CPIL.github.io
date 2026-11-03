"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _compute_circumradii(points):
    """
    Compute the circumradii of the facets of the convex hull of the points.
    Returns (radii, is_valid) where is_valid is True if origin is inside the hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), False

    # Check if origin is inside the convex hull
    # hull.equations: shape (n_facets, 4). [a, b, c, d] such that ax+by+cz=d
    # Normals are normalized to unit length in scipy.
    # If origin is inside, d should be positive for all facets (distance from origin to plane).
    # If any d is negative, origin is outside.
    d_vals = hull.equations[:, 3]
    if np.min(d_vals) <= 1e-10:
        return np.array([]), False

    facets = hull.simplices
    radii = np.zeros(len(facets))
    
    for i, tri in enumerate(facets):
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        # Compute normal to the plane of the triangle
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            # Degenerate triangle
            radii[i] = np.pi / 2.0
            continue
            
        normal = normal / norm_normal
        
        # Distance from origin to the plane
        d = np.dot(p1, normal)
        
        # Center of the circle in R3
        c_euclid = d * normal
        c_norm = np.linalg.norm(c_euclid)
        
        if c_norm < 1e-10:
            # Plane passes through origin, circumradius is 90 degrees
            radii[i] = np.pi / 2.0
            continue
            
        c_spher = c_euclid / c_norm
        
        # Circumradius is the angle between c_spher and p1
        dot_val = np.dot(c_spher, p1)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    return radii, True


def _objective(params):
    """
    Objective function: Sum of squared circumradii of the convex hull facets.
    Penalize if origin is not inside the hull.
    """
    points = params.reshape(N, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    radii, is_valid = _compute_circumradii(points)
    
    if not is_valid:
        return 1000.0
        
    # Sum of squares is a smooth proxy for max radius
    return np.sum(radii ** 2)


def _local_optimize(points):
    """
    Optimize the configuration using BFGS on the covering radius proxy.
    """
    x0 = points.flatten()
    result = minimize(_objective, x0, method='BFGS', options={'maxiter': 200, 'disp': False})
    best_points = result.x.reshape(N, 3)
    return best_points / np.linalg.norm(best_points, axis=1, keepdims=True)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    best_score = np.inf
    best_points = None
    
    # Multiple random restarts to avoid local minima
    for _ in range(5):
        # Random initialization
        rng = np.random.default_rng()
        x0 = rng.standard_normal((N, 3))
        x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        
        # Optimize
        points_opt = _local_optimize(x0)
        
        # Evaluate final score (Covering Radius in degrees)
        radii, is_valid = _compute_circumradii(points_opt)
        if not is_valid:
            continue
            
        max_radius_deg = np.max(radii) * 180.0 / np.pi
        
        if max_radius_deg < best_score:
            best_score = max_radius_deg
            best_points = points_opt
            
    if best_points is None:
        # Fallback to Fibonacci if all fail
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
# EVOLVE-BLOCK-END
