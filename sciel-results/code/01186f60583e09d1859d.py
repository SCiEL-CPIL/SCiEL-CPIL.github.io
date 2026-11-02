"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _spherical_circumradius(p1, p2, p3):
    """
    Compute the spherical circumradius (angular distance) of the triangle formed by
    three points on the unit sphere. This is the angular distance from the spherical
    circumcenter to any of the vertices.
    
    The spherical circumcenter is the intersection of the sphere with the plane
    perpendicular to the normal of the triangle passing through the centroid? No.
    The spherical circumcenter is the point on the sphere equidistant from p1, p2, p3.
    It lies along the normal vector of the triangle formed by p1, p2, p3 in 3D space.
    
    Let n = normalize(p1 x p2 + p2 x p3 + p3 x p1) ? No.
    The normal vector to the plane of the triangle is n = (p2 - p1) x (p3 - p1).
    The spherical circumcenter c is along the direction of the vector:
    v = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)
    Actually, a simpler way: The circumcenter c is the normalized vector:
    c = (p1 * |p2-p3|^2 + p2 * |p3-p1|^2 + p3 * |p1-p2|^2) / |...| ?
    
    Let's use the property that c is equidistant from p1, p2, p3.
    c = alpha * p1 + beta * p2 + gamma * p3.
    Due to symmetry, if the triangle is not degenerate, the circumcenter lies on the line
    passing through the origin and the 3D circumcenter of the triangle p1,p2,p3? No, that's for the Euclidean circumcenter.
    
    The spherical circumcenter is the normalization of the vector:
    w = p1 * (p2 . p3) + p2 * (p3 . p1) + p3 * (p1 . p2) ? No.
    
    Correct formula for spherical circumcenter direction:
    The normal to the plane of the triangle is n = (p2 - p1) x (p3 - p1).
    The circumcenter c is in the direction of n? No, only if the triangle is right-angled or something.
    
    Actually, the spherical circumcenter is the point on the sphere that is the intersection of the sphere and the line passing through the origin and the Euclidean circumcenter of the triangle p1, p2, p3? 
    No, the Euclidean circumcenter of the triangle formed by points on the sphere is NOT generally on the line from the origin to the spherical circumcenter.
    
    However, there is a known formula. The spherical circumradius R is given by:
    R = arccos( (1 - d12^2 - d23^2 - d31^2 + 2*d12*d23*d31) / (2 * (1 - ...)) )? Too complex.
    
    Simpler: The spherical circumcenter is the normalized vector:
    c = (p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)) / norm(...)
    Let's verify. If p1, p2, p3 are close, this approximates the normal.
    
    Actually, the vector v = p1 x p2 + p2 x p3 + p3 x p1 is the normal to the plane? No.
    The normal is n = (p2 - p1) x (p3 - p1).
    
    Let's use the fact that the covering radius is the max angular distance.
    For a triangle on the sphere, the farthest point from the vertices is the spherical circumcenter.
    The angular distance is the spherical circumradius.
    
    We can compute the spherical circumradius R using the side lengths a, b, c (angular distances).
    Using spherical law of cosines:
    cos a = cos b cos c + sin b sin c cos A
    We can find the angles A, B, C. Then R = ?
    
    Alternatively, we can just compute the Euclidean circumradius of the triangle p1, p2, p3.
    Let r_euc be the Euclidean circumradius of the triangle in 3D.
    The distance from the origin to the plane of the triangle is h.
    The distance from the origin to the Euclidean circumcenter is d.
    Then the spherical circumradius R satisfies cos(R) = h / 1 ? No.
    
    Let's use a robust numerical method:
    The spherical circumcenter is the minimizer of max(d(c, p1), d(c, p2), d(c, p3)).
    But we can compute it analytically.
    The direction of the circumcenter is:
    u = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)
    This vector u is perpendicular to the plane of the triangle?
    u . (p2 - p1) = p1 . (p2 x p3) - p1 . (p1 x p3) + p2 . (p2 x p3) - p2 . (p1 x p3) + p3 . (p1 x p2) - p3 . (p1 x p3)
    This is getting messy.
    
    Let's use a simpler, robust approach for the objective:
    The covering radius is the max over all points on the sphere.
    The maximum occurs at a vertex of the Voronoi diagram, which is the circumcenter of a Delaunay triangle.
    So, we compute the convex hull.
    For each facet (triangle) of the convex hull, we compute the spherical circumradius.
    
    Formula for spherical circumradius R of a triangle with vertices on unit sphere:
    Let a, b, c be the side lengths (angular).
    Let s = (a+b+c)/2.
    Area K = sqrt(sin(s) sin(s-a) sin(s-b) sin(s-c)).
    R = sqrt( (1/4) * (1/sin^2(a/2) + 1/sin^2(b/2) + 1/sin^2(c/2)) )? No.
    
    Actually, R = arccos( (1 - a^2 - b^2 - c^2 + 2abc) / (2(1 - ...)) ) is for something else.
    
    Let's use the vector formula:
    The spherical circumcenter c is given by:
    c = normalize( p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2) )
    This is the vector pointing to the circumcenter.
    Then R = arccos(c . p1).
    
    Let's test this mentally. If p1, p2, p3 form a small triangle, p1 x p2 is normal.
    This formula is standard for the circumcenter of a triangle on a sphere.
    
    If the points are collinear or nearly so, this might be unstable.
    """
    # Compute the vector to the circumcenter
    # v = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)
    v = np.cross(p2, p3) * p1 + np.cross(p3, p1) * p2 + np.cross(p1, p2) * p3
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        # Degenerate triangle, return a large value or estimate
        return np.pi / 2 
    c = v / norm_v
    # The angular distance is the arccos of the dot product with any vertex
    # Use p1
    dot = np.dot(c, p1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(points):
    """
    Compute the covering radius by finding the max spherical circumradius
    of the triangular facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If convex hull fails, fallback to sampling
        n_samples = 10000
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        phi = np.arccos(np.random.uniform(-1, 1, n_samples))
        samples = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        dots = np.dot(samples, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(min_dots)
        return np.max(angles)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        r = _spherical_circumradius(p1, p2, p3)
        if r > max_radius:
            max_radius = r
            
    return max_radius

def _params_to_points(params):
    """Convert 3N parameters to N points on the unit sphere."""
    params = params.reshape(N, 3)
    norms = np.linalg.norm(params, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return params / norms

def _objective(params):
    """Objective function: covering radius in radians."""
    points = _params_to_points(params)
    return _covering_radius(points)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    np.random.seed(42)
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points0 = points0 / np.linalg.norm(points0, axis=1, keepdims=True)
    
    params0 = points0.flatten()
    
    # Use Nelder-Mead for local optimization of the deterministic covering radius
    # Nelder-Mead is robust for non-smooth functions like max-circumradius
    res = minimize(
        _objective,
        params0,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )
    
    best_params = res.x
    best_val = res.fun
    
    # Try a few random restarts to avoid local minima
    for restart in range(5):
        # Perturb the initial points
        noise = np.random.normal(0, 0.5, (N, 3))
        points_init = points0 + noise
        points_init = points_init / np.linalg.norm(points_init, axis=1, keepdims=True)
        params_init = points_init.flatten()
        
        res_r = minimize(
            _objective,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res_r.fun < best_val:
            best_val = res_r.fun
            best_params = res_r.x
            
    points = _params_to_points(best_params)
    return points
# EVOLVE-BLOCK-END
