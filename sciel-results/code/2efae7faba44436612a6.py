import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _spherical_circumcenters(points, facets):
    """
    Compute circumcenters of spherical triangles defined by points and facets.
    Returns an array of shape (n_facets, 3) with normalized circumcenter directions.
    """
    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # The circumcenter of a spherical triangle is the intersection of the sphere
    # with the line passing through the origin and the vector a + b + c (if the triangle is small)
    # More robustly, it is the normalized vector (a x b) x (b x c) + ... ?
    # Actually, the circumcenter c satisfies c . a = c . b = c . c.
    # For unit vectors, the circumcenter is proportional to a + b + c only if the triangle is equilateral? No.
    # The correct formula for the circumcenter of a triangle with vertices a, b, c on the unit sphere is:
    # c = (a * (sin(2*B) + sin(2*C)) + b * (sin(2*A) + sin(2*C)) + c * (sin(2*A) + sin(2*B))) / (2 * sin(A) * sin(B) * sin(C)) ?
    # This is complicated.
    
    # A simpler and robust method:
    # The circumcenter lies in the plane defined by a, b, c.
    # It is also equidistant from a, b, c.
    # Let n = a x b. The plane containing a, b, c has normal m = (b-a) x (c-a).
    # The circumcenter is the intersection of the sphere and the line defined by the cross product of the normals?
    
    # Actually, a very robust method is to solve for the circumcenter as the point on the sphere
    # that is equidistant from a, b, c.
    # Let x be the circumcenter. x . a = x . b = x . c = cos(R).
    # This implies x is in the null space of [a; b; c] - cos(R) * 1? No.
    
    # Let's use the formula:
    # c = (a * (|b|^2 |c|^2 (1 - (b . c)^2)) + ... )
    
    # A known robust formula for the circumcenter of a spherical triangle with vertices a, b, c (unit vectors) is:
    # c = (a * (sin(2 * angle_b) + sin(2 * angle_c)) + b * (sin(2 * angle_a) + sin(2 * angle_c)) + c * (sin(2 * angle_a) + sin(2 * angle_b)))
    # This is still complex.
    
    # Let's use the property that the circumcenter is the normalized vector:
    # c = (a x b) x (b x c) + (b x c) x (c x a) + (c x a) x (a x b)
    # This vector is perpendicular to the plane of the triangle and points to the circumcenter?
    # Actually, the vector n = (a x b) x (b x c) + (b x c) x (c x a) + (c x a) x (a x b) is the normal to the plane? No.
    
    # Let's use a simpler approach:
    # The circumcenter is the intersection of the sphere and the line L(t) = t * (a + b + c) ? No, only for equilateral.
    
    # Correct robust method:
    # The circumcenter c satisfies:
    # c . a = c . b = c . c
    # c = alpha * a + beta * b + gamma * c
    # This is a linear system.
    # But c must be on the unit sphere.
    
    # Let's use the formula from "Spherical Geometry" by K. S. Maslov:
    # The circumcenter is given by:
    # c = (a * (sin(2*B) + sin(2*C)) + b * (sin(2*A) + sin(2*C)) + c * (sin(2*A) + sin(2*B))) / (2 * sin(A) * sin(B) * sin(C))
    # where A, B, C are the angles of the spherical triangle.
    
    # This is too slow to compute for every facet.
    
    # Alternative:
    # The circumcenter is the normalized vector:
    # c = (a x b) x (b x c) + (b x c) x (c x a) + (c x a) x (a x b)
    # Let's test this.
    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    
    # The vector sum of these cross products is the normal to the plane?
    # Actually, the vector n = ab + bc + ca is not the circumcenter.
    
    # Let's use the fact that the circumcenter is the intersection of the sphere and the line
    # defined by the cross product of the normals of the great circles?
    
    # A very robust and simple method:
    # The circumcenter is the normalized vector:
    # c = (a * (b . c) + b * (c . a) + c * (a . b) - (a . b * b . c * c . a) * (a + b + c)) / ... ?
    
    # Let's use the formula:
    # c = (a * (sin(2 * angle_b) + sin(2 * angle_c)) + ...)
    
    # Actually, the simplest robust method is to use the fact that the circumcenter is the point on the sphere
    # that is equidistant from a, b, c.
    # This is the same as saying that the circumcenter is the intersection of the sphere and the line
    # that is the intersection of the planes:
    # x . (a - b) = 0
    # x . (b - c) = 0
    # So x is the cross product of (a - b) and (b - c).
    # Let's try this.
    
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norms_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    valid = norms_n.squeeze() > 1e-12
    centers = np.zeros_like(a)
    
    if np.any(valid):
        centers[valid] = n[valid] / norms_n[valid]
        
    return centers

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        if len(facets) == 0:
            return np.inf
            
        centers = _spherical_circumcenters(points, facets)
        
        a = points[facets[:, 0]]
        cos_R = np.sum(a * centers, axis=1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        R = np.arccos(cos_R)
        return np.degrees(np.max(R))
    except Exception:
        return np.inf

def _objective(flat):
    """
    Objective function for optimization.
    """
    pts = flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    valid = norms.squeeze() > 1e-12
    if not np.all(valid):
        return np.inf
    pts[valid] /= norms[valid]
    return _covering_radius(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def fibonacci_spiral(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    best_points = fibonacci_spiral(N)
    best_radius = _covering_radius(best_points)
    
    np.random.seed(42)
    
    # Multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Different initializations
        if restart == 0:
            init = best_points.copy()
        elif restart == 1:
            init = fibonacci_spiral(N, offset=0.1)
        elif restart == 2:
            init = fibonacci_spiral(N, offset=0.9)
        else:
            init = np.random.randn(N, 3)
            init /= np.linalg.norm(init, axis=1, keepdims=True)
        
        # Optimize from this start
        time_budget = min(20.0, deadline - 30.0 - (time.time() - start_time))
        if time_budget < 2.0:
            break
            
        try:
            res = minimize(
                _objective, 
                init.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_radius:
                best_radius = res.fun
                best_points = res.x.reshape(N, 3)
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points /= norms
        except Exception:
            pass
    
    # Final safety normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points /= norms
    
    return best_points
