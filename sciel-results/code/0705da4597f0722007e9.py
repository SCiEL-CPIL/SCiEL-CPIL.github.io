"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _spherical_circumradius(a, b, c):
    """Compute the spherical circumradius for a spherical triangle with vertices a, b, c (unit vectors).
    Returns (radius, center) where center is the unit vector of the circumcenter (outward)."""
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return np.pi, None
    
    n = n / n_norm
    
    dot_a = np.dot(n, a)
    dot_b = np.dot(n, b)
    dot_c = np.dot(n, c)
    
    if dot_a > 0 and dot_b > 0 and dot_c > 0:
        center = n
    else:
        center = -n
    
    d = np.dot(center, a)
    d = np.clip(d, -1.0, 1.0)
    radius = np.arccos(d)
    
    return radius, center


def _covering_radius_and_worst(points):
    """Compute the covering radius and identify the worst facet.
    Returns (max_radius, worst_simplex, worst_center)."""
    try:
        tri = Delaunay(points)
    except Exception:
        return np.pi, None, None
    
    max_r = 0.0
    worst_simplex = None
    worst_center = None
    
    for simplex in tri.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        r, center = _spherical_circumradius(a, b, c)
        if center is not None and r > max_r:
            max_r = r
            worst_simplex = simplex
            worst_center = center
    
    return max_r, worst_simplex, worst_center


def _move_point_geodesic(p, direction, step):
    """Move point p along geodesic in direction 'direction' by 'step' radians."""
    new_p = p + step * direction
    norm = np.linalg.norm(new_p)
    if norm < 1e-12:
        return p
    return new_p / norm


def _tangent_direction(p, target):
    """Get the unit tangent direction at p pointing toward target."""
    proj = target - np.dot(target, p) * p
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        return np.zeros(3)
    return proj / norm


def _local_refinement(points, max_iters=25):
    """Perform local refinement by targeting the worst facet and optimizing its vertices."""
    points = points.copy()
    
    for iteration in range(max_iters):
        max_r, worst_simplex, worst_center = _covering_radius_and_worst(points)
        
        if worst_simplex is None:
            break
        
        idx_a, idx_b, idx_c = worst_simplex
        a = points[idx_a].copy()
        b = points[idx_b].copy()
        c = points[idx_c].copy()
        
        # Compute tangent bases for each vertex
        def make_tangent_basis(p):
            # Find two orthonormal vectors in the tangent plane at p
            if abs(p[2]) < 0.9:
                e1 = np.array([1.0, 0.0, 0.0])
            else:
                e1 = np.array([0.0, 1.0, 0.0])
            t1 = e1 - np.dot(e1, p) * p
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(p, t1)
            t2 = t2 / np.linalg.norm(t2)
            return t1, t2
        
        t1a, t2a = make_tangent_basis(a)
        t1b, t2b = make_tangent_basis(b)
        t1c, t2c = make_tangent_basis(c)
        
        # Objective: minimize circumradius of the facet, parameterized by 6 tangent coordinates
        def objective(params):
            ua, va = params[0], params[1]
            ub, vb = params[2], params[3]
            uc, vc = params[4], params[5]
            
            # New positions via geodesic movement
            new_a = _move_point_geodesic(a, ua * t1a + va * t2a, 1.0)
            new_b = _move_point_geodesic(b, ub * t1b + vb * t2b, 1.0)
            new_c = _move_point_geodesic(c, uc * t1c + vc * t2c, 1.0)
            
            r, _ = _spherical_circumradius(new_a, new_b, new_c)
            return r
        
        # Initial guess: small movement toward circumcenter
        dir_a = _tangent_direction(a, worst_center)
        dir_b = _tangent_direction(b, worst_center)
        dir_c = _tangent_direction(c, worst_center)
        
        x0 = np.array([
            np.dot(dir_a, t1a), np.dot(dir_a, t2a),
            np.dot(dir_b, t1b), np.dot(dir_b, t2b),
            np.dot(dir_c, t1c), np.dot(dir_c, t2c)
        ]) * 0.01  # small initial step
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            bounds=[(-0.1, 0.1)] * 6,
                            options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if result.success and result.fun < max_r - 1e-10:
                ua, va, ub, vb, uc, vc = result.x
                points[idx_a] = _move_point_geodesic(a, ua * t1a + va * t2a, 1.0)
                points[idx_b] = _move_point_geodesic(b, ub * t1b + vb * t2b, 1.0)
                points[idx_c] = _move_point_geodesic(c, uc * t1c + vc * t2c, 1.0)
            else:
                # Fallback: small step toward circumcenter
                step = 0.005
                points[idx_a] = _move_point_geodesic(a, dir_a, step)
                points[idx_b] = _move_point_geodesic(b, dir_b, step)
                points[idx_c] = _move_point_geodesic(c, dir_c, step)
        except Exception:
            step = 0.005
            points[idx_a] = _move_point_geodesic(a, dir_a, step)
            points[idx_b] = _move_point_geodesic(b, dir_b, step)
            points[idx_c] = _move_point_geodesic(c, dir_c, step)
    
    return points


def construct():
    """Construct 50 points on the unit sphere with minimal covering radius."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    points = _local_refinement(points, max_iters=25)
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
