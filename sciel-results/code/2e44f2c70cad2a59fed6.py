"""Improved program for covering the sphere with 50 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in pts. This equals the largest circumradius of the spherical
    Voronoi cells, which corresponds to the largest circumradius of the facets of the
    convex hull (when all points are in convex position).
    
    We compute the convex hull and for each facet, find the circumcenter of the
    triangle (for 3D hull facets which are triangles). The angular distance from
    the facet's circumcenter (projected to sphere) to any vertex of the facet gives
    the local covering radius for that cell.
    """
    pts = _normalize(pts)
    
    # Check if all points are in a closed hemisphere
    # If they are, the covering radius is at least 90 degrees
    # We can check by seeing if there exists a plane through origin such that all points are on one side
    # More simply: if the origin is not inside the convex hull, they're in a hemisphere
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # For each facet (triangle), compute the circumradius
    # The facet is defined by 3 points. The circumcenter of the triangle in 3D
    # projected onto the sphere gives the point farthest from the facet vertices.
    max_radius_deg = 0.0
    
    for simplex in hull.simplices:
        # simplex is an array of 3 indices for a triangular facet
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        # Compute the circumcenter of the triangle p0, p1, p2
        # The circumcenter is equidistant from all three vertices.
        # In 3D, the circumcenter of a triangle lies on the line perpendicular to the
        # triangle's plane passing through the circumcenter of the 2D triangle.
        # For points on the unit sphere, the circumcenter of the spherical triangle
        # is the normalized cross product direction... actually, let's compute it properly.
        
        # The circumcenter C satisfies |C - p0| = |C - p1| = |C - p2|
        # and C lies in the plane of the triangle.
        
        # Method: find the circumcenter of the 2D triangle.
        # Let a = p1 - p0, b = p2 - p0
        # The circumcenter in the plane is p0 + s*a + t*b where:
        # |p0 + s*a + t*b - p0|^2 = |p0 + s*a + t*b - p1|^2
        # |s*a + t*b|^2 = |s*a + t*b - a|^2
        # 0 = -2*s*a·a - 2*t*b·a + |a|^2
        # Similarly: 0 = -2*s*a·b - 2*t*b·b + |b|^2
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            # Degenerate triangle
            continue
            
        s = (bb * ab - ab * bb) / (2 * denom)  # Wait, let me redo this
        # From the two equations:
        # 2*s*aa + 2*t*ab = aa  =>  s*aa + t*ab = aa/2
        # 2*s*ab + 2*t*bb = bb  =>  s*ab + t*bb = bb/2
        
        # Solving:
        # s = (aa/2 * bb - ab/2 * ab) / denom = (aa*bb - ab^2) / (2*denom)
        # t = (aa/2 * ab - aa/2 * ab)... let me use Cramer's rule
        # s = (aa/2 * bb - ab/2 * ab) / denom = (aa*bb - ab^2) / (2*denom)
        # t = (aa * ab/2 - ab/2 * aa) / denom... 
        # Actually:
        # s = ( (aa/2)*bb - (ab/2)*ab ) / denom = (aa*bb - ab^2) / (2*denom)
        # t = ( aa*(ab/2) - (aa/2)*aa ) / denom = (aa*ab - aa^2) / (2*denom)
        
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        
        # The circumradius is the distance from circumcenter to any vertex
        R = np.linalg.norm(circumcenter_2d - p0)
        
        # The angular radius on the sphere: the circumcenter_2d is in the plane of the triangle.
        # The point on the sphere that is the circumcenter of the spherical triangle is
        # circumcenter_2d / |circumcenter_2d| (if it's not the origin).
        # The angular distance from this point to p0 is arcsin(R / |circumcenter_2d|)... 
        # Actually, if C = circumcenter_2d, then |C| is the distance from origin to the circumcenter.
        # The angular distance theta satisfies: cos(theta) = (C · p0) / |C|
        # And |C - p0| = R, so |C|^2 - 2*C·p0 + |p0|^2 = R^2
        # |C|^2 - 2*C·p0 + 1 = R^2
        # C·p0 = (|C|^2 + 1 - R^2) / 2
        # cos(theta) = (|C|^2 + 1 - R^2) / (2 * |C|)
        
        C_norm = np.linalg.norm(circumcenter_2d)
        if C_norm < 1e-15:
            # Circumcenter at origin, meaning the triangle spans a hemisphere
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
    
    return max_radius_deg


def construct():
    """
    Use a multi-start local optimization approach to find a good covering configuration.
    We start with several initial configurations (including Fibonacci spiral and random starts)
    and apply gradient-free optimization (Nelder-Mead or similar) to minimize the covering radius.
    """
    best_pts = None
    best_radius = np.inf
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(_normalize(fib))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    
    # 3. Start from known good structures: icosahedron + perturbations
    # Icosahedron has 12 vertices. We can start with icosahedron and add points.
    # Actually, let's just use the Fibonacci and random starts with optimization.
    
    from scipy.optimize import minimize
    
    def objective(params):
        """Flatten/unflatten and compute covering radius."""
        pts = _normalize(params.reshape(N, 3))
        return _covering_radius(pts)
    
    for init in inits:
        try:
            result = minimize(
                objective,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            pts = _normalize(result.x.reshape(N, 3))
            radius = _covering_radius(pts)
            if radius < best_radius:
                best_radius = radius
                best_pts = pts
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _normalize(inits[0])
    
    return best_pts
