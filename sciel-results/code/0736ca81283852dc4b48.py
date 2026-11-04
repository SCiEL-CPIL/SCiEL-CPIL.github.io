"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees.
    
    The covering radius is the max angular distance from any point on the sphere
    to the nearest configuration point. This equals the largest circumradius of
    the facets of the convex hull (each facet's circumscribed circle on the sphere).
    
    We compute this by finding, for each point, the angular distances to its
    nearest neighbors, and then finding the 'worst' point on the sphere.
    
    A practical approach: the covering radius is determined by the points that are
    equidistant to 3 or more configuration points (vertices of the Voronoi cells
    on the sphere). We approximate by checking circumcircles of triangles formed
    by nearby points.
    """
    pts = _normalize(pts)
    # Compute pairwise angular distances
    # cos of angular distance = dot product
    dot = pts @ pts.T
    np.fill_diagonal(dot, 2.0)  # avoid self
    dot = np.clip(dot, -1.0, 1.0)
    ang_dist = np.arccos(dot)  # radians
    
    # For each point, find its k nearest neighbors
    k = 8
    nearest_idx = np.argsort(ang_dist, axis=1)[:, :k]
    
    # For each point, find the worst case: the point on the sphere farthest from all
    # We use the fact that the covering radius is the max over all points of min distance.
    # The worst points are at the circumcenters of triangles of mutually near points.
    
    # Simpler robust approach: sample the sphere densely and compute min distance
    # But that's expensive. Instead, use the fact that the covering radius
    # equals the largest value r such that there exists a point at distance >= r from all pts.
    
    # Use Delaunay-like approach: for each triple of points that are mutually close,
    # compute the circumradius of the spherical triangle.
    
    # Alternative: use scipy's ConvexHull on the 3D points, then for each face,
    # compute the circumradius of the spherical triangle.
    
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: dense sampling
        theta = np.linspace(0, np.pi, 300)
        phi = np.linspace(0, 2*np.pi, 600)
        TH, PH = np.meshgrid(theta, phi)
        sx = np.sin(TH) * np.cos(PH)
        sy = np.sin(TH) * np.sin(PH)
        sz = np.cos(TH)
        samples = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()])
        min_dists = np.arccos(np.clip(np.max(pts @ samples.T, axis=0), -1.0, 1.0))
        return np.degrees(np.max(min_dists))
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Each simplex is a triangle (3 points) on the sphere
        p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Check if the circumcenter of this spherical triangle is on the sphere
        # and compute its distance to the three vertices
        # The circumcenter in 3D of the triangle formed by p0, p1, p2
        # lies along the normal to the plane containing p0, p1, p2
        
        # Normal to the plane
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter C satisfies |C - p0| = |C - p1| = |C - p2|
        # and C = p0 + a*v1 + b*v2 for some a, b
        # Or: C lies on the line through the origin and n (if the triangle is acute)
        # Actually, the circumcenter of the triangle on the sphere is the point
        # equidistant from p0, p1, p2 on the sphere.
        
        # Solve for C = t * n such that C is on the unit sphere and equidistant from p0, p1, p2
        # |C - p0|^2 = |C - p1|^2
        # |C|^2 - 2*C.p0 + |p0|^2 = |C|^2 - 2*C.p1 + |p1|^2
        # Since |C| = 1 and |p0| = |p1| = 1:
        # -2*C.p0 = -2*C.p1 => C.p0 = C.p1
        # Similarly C.p0 = C.p2
        
        # C = t*n, so t*n.p0 = t*n.p1 => n.p0 = n.p1 (should be true if n is perpendicular to p1-p0)
        # The circumcenter on the sphere is at the intersection of the sphere with the line
        # that is the perpendicular bisector of the chord... 
        
        # Better: the circumradius of the spherical triangle is the angular radius
        # of the smallest spherical cap containing the three points.
        # This is the angle between the circumcenter and any vertex.
        
        # Compute circumcenter as the normalized sum of points weighted by opposite side lengths
        # Or use the formula: circumcenter = normalize(p0 * sin(2*A) + p1 * sin(2*B) + p2 * sin(2*C))
        # where A, B, C are the angles of the spherical triangle at each vertex.
        
        # Simpler: find the point on the sphere that maximizes the minimum distance to p0, p1, p2
        # This is the center of the smallest enclosing cap.
        
        # Use the fact that for three points on the unit sphere, the circumcenter
        # (if the triangle is acute) is proportional to the cross product approach.
        
        # Let's compute it directly: we want C on unit sphere such that C.p0 = C.p1 = C.p2 = cos(r)
        # C = alpha * n + beta * (p0 + p1 + p2) ... this is getting complex.
        
        # Direct approach: the circumradius r satisfies:
        # cos(r) = C.p0 for some unit C
        # The minimum enclosing circle of three points on the sphere:
        # If the triangle is acute, r = circumradius of the spherical triangle
        # Otherwise, r = half the largest side
        
        a = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))  # side opposite p0
        b = np.arccos(np.clip(np.dot(p0, p2), -1.0, 1.0))  # side opposite p1
        c = np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0))  # side opposite p2
        
        # Spherical law of cosines to check if acute
        # cos(A) = (cos(a) - cos(b)*cos(c)) / (sin(b)*sin(c))
        sin_b = np.sin(b)
        sin_c = np.sin(c)
        sin_a = np.sin(a)
        sin_b2 = np.sin(b)
        sin_c2 = np.sin(c)
        
        if sin_b * sin_c < 1e-12:
            # Degenerate
            r = max(a, b, c) / 2
        else:
            cos_A = (np.cos(a) - np.cos(b) * np.cos(c)) / (sin_b * sin_c)
            cos_A = np.clip(cos_A, -1.0, 1.0)
            A = np.arccos(cos_A)
            
            if A < np.pi / 2:
                # Acute at A, compute circumradius using spherical law of sines
                # sin(r) = sin(a) / (2 * sin(A)) ... no, that's for Euclidean
                # For spherical: the circumradius R satisfies
                # sin(a) = 2 * sin(R) * sin(A) ... let me use a different approach
                
                # The circumcenter C satisfies C.p0 = C.p1 = C.p2 = cos(R)
                # C is along the normal n to the plane of the triangle (for acute case)
                # C = n / |n| (already normalized)
                # cos(R) = C.p0 = n.p0
                cos_R = np.dot(n, p0)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                
                # Verify: check that C is on the same side
                if np.dot(n, p0) < 0:
                    n = -n
                    cos_R = np.dot(n, p0)
                    cos_R = np.clip(cos_R, -1.0, 1.0)
                    R = np.arccos(cos_R)
                
                r = R
            else:
                # Obtuse, the smallest enclosing cap is determined by the longest side
                r = max(a, b, c) / 2
        
        max_radius = max(max_radius, r)
    
    return np.degrees(max_radius)


def _fast_covering_radius(pts):
    """Faster approximation of covering radius for optimization."""
    pts = _normalize(pts)
    dot = pts @ pts.T
    np.fill_diagonal(dot, 2.0)
    dot = np.clip(dot, -1.0, 1.0)
    ang_dist = np.arccos(dot)
    
    # For each point, the worst-case distance in its Voronoi cell
    # Approximate by finding the point on the sphere farthest from all pts
    # Use a grid-based approach with adaptive refinement
    
    # First pass: coarse grid
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    sx = np.sin(TH) * np.cos(PH)
    sy = np.sin(TH) * np.sin(PH)
    sz = np.cos(TH)
    samples = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()])
    
    # Min distance from each sample to nearest point
    dots = pts @ samples.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=0)  # max dot = min angle
    min_angles = np.arccos(min_dots)
    
    return np.degrees(np.max(min_angles))


def _objective(x):
    """Objective function for optimization."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    return _fast_covering_radius(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = pts.copy()
    best_score = _covering_radius_deg(best_pts)
    
    # Local optimization using L-BFGS-B on the sphere
    # Parameterize each point by (theta, phi) or use tangent space
    
    # Use scipy's minimize with multiple restarts
    from scipy.optimize import minimize
    
    def _local_optimize(pts0, time_budget):
        """Local optimization using gradient descent on the sphere."""
        pts = _normalize(pts0.copy())
        
        # Use BFGS on the flattened coordinates with projection
        def obj(x):
            p = _normalize(x.reshape(N, 3))
            return _fast_covering_radius(p)
        
        def grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                g[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
            return g
        
        result = minimize(obj, pts.ravel(), method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        return _normalize(result.x.reshape(N, 3)), result.fun
    
    # Do a few local optimizations from the initial point
    for iteration in range(3):
        if time.time() > deadline - 30:
            break
        
        pts, score = _local_optimize(best_pts, deadline)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Try differential evolution for global optimization
    if time.time() < deadline - 60:
        # Parameterize as (theta, phi) for each point
        def _to_sphere(x):
            thetas = x[:N]
            phis = x[N:]
            sx = np.sin(thetas) * np.cos(phis)
            sy = np.sin(thetas) * np.sin(phis)
            sz = np.cos(thetas)
            return np.column_stack([sx, sy, sz])
        
        def _obj_sphere(x):
            p = _to_sphere(x)
            return _fast_covering_radius(p)
        
        bounds = [(0, np.pi)] * N + [(0, 2*np.pi)] * N
        
        # Use differential evolution with limited iterations
        try:
            de_result = differential_evolution(_obj_sphere, bounds, 
                                                maxiter=50, popsize=15,
                                                seed=42, tol=1e-8,
                                                init='sobol')
            if de_result.fun < best_score:
                best_pts = _to_sphere(de_result.x)
                best_score = de_result.fun
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 10:
        pts, score = _local_optimize(best_pts, deadline)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
