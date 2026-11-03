"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius of points on the unit sphere.
    Uses the convex hull to find facets, then checks the circumradius
    of each facet (only if the cap is empty, i.e., the facet is on the hull).
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # Get the facets (simplices) of the hull
    simplices = hull.simplices
    
    # For each facet, compute the circumradius
    # The circumradius of a simplex on the unit sphere corresponds to the
    # angular distance from the sphere center to the farthest point on the cap
    # Actually, we need the maximum angular distance from any point on the sphere
    # to its nearest configuration point.
    
    # A more direct approach: the covering radius is the maximum over all points
    # on the sphere of the minimum angular distance to any configuration point.
    # This equals the maximum circumradius of the Delaunay triangulation facets
    # that have empty caps.
    
    # Alternative: use the fact that the covering radius is determined by the
    # Voronoi diagram on the sphere. The critical points are the circumcenters
    # of the Delaunay triangles (facets of the convex hull that are visible from
    # the origin, i.e., whose normal points away from the origin).
    
    # For a set of points on the unit sphere, the covering radius is the maximum
    # angular distance from any point on the sphere to its nearest site.
    # This is achieved at a circumcenter of some empty circle (on the sphere).
    
    # Practical approach: sample the sphere densely and compute max min-distance.
    # But this is expensive. Better: use the convex hull facets.
    
    # The key insight: for points on the unit sphere, the covering radius equals
    # the maximum over all facets of the convex hull of the circumradius of that
    # facet, but only for facets whose outward normal (from the origin) has a
    # positive dot product with the facet center (i.e., the facet is "visible" 
    # from outside, meaning the cap it bounds is empty).
    
    # Actually, for the convex hull of points on the unit sphere, all facets are
    # on the boundary. The covering radius is the maximum angular radius of the
    # spherical cap centered at a facet's circumcenter that contains no other points.
    
    # Let's use a different, more robust approach:
    # The covering radius is the smallest r such that spherical caps of radius r
    # centered at the points cover the sphere.
    # This is equivalent to: max_{y in S^2} min_i angle(y, p_i)
    
    # We can compute this by finding the maximum over the vertices of the
    # Voronoi diagram, which are the circumcenters of Delaunay triangles.
    
    # For the convex hull in 3D, the Delaunay triangulation of points on the
    # sphere corresponds to the faces of the convex hull.
    
    # Let's compute for each facet (triangle) the circumcenter on the sphere
    # and the angular distance from that circumcenter to the three vertices.
    # The covering radius is the maximum of these over all facets.
    
    max_radius = 0.0
    
    for simplex in simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of the triangle on the unit sphere.
        # The circumcenter is the point equidistant from all three vertices.
        # For a triangle on the unit sphere, the circumcenter lies along the
        # line from the origin through the centroid of the triangle's normal.
        
        # The circumcenter c satisfies |c| = 1 and c·v0 = c·v1 = c·v2 = cos(r)
        # where r is the circumradius (angular).
        
        # We can find c by solving the linear system:
        # c·v0 = c·v1 = c·v2
        # |c| = 1
        
        # c is perpendicular to (v0 - v1) and (v0 - v2), so c is parallel to
        # the cross product (v1 - v0) × (v2 - v0).
        
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-12:
            continue
        
        c = n / n_norm
        
        # The circumcenter could be c or -c. We need the one that is "farthest"
        # from the origin in the sense of the empty cap. Actually, both are on
        # the unit sphere. The correct circumcenter is the one that is on the
        # same side as the facet's outward normal.
        
        # The outward normal of the facet (pointing away from the origin) is
        # the direction of the centroid of the three vertices (since they're on
        # the unit sphere, the centroid points in the direction of the facet).
        
        centroid = (v0 + v1 + v2) / 3.0
        centroid_norm = np.linalg.norm(centroid)
        
        if centroid_norm < 1e-12:
            continue
        
        # The facet is "visible" from the origin if the centroid and the normal
        # n point in roughly the same direction.
        if np.dot(c, centroid) < 0:
            c = -c
        
        # Now compute the angular distance from c to v0 (should be same for v1, v2)
        dot_val = np.clip(np.dot(c, v0), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_radius:
            max_radius = r
    
    return max_radius

def _objective_flat(x, pts_fixed=None):
    """
    Objective function for optimization.
    We maximize the minimum pairwise distance (or minimize the covering radius).
    Here we use a soft-max approximation of the covering radius.
    """
    pts = _normalize(x.reshape(N, 3))
    
    # Compute all pairwise angular distances
    # dist[i,j] = arccos(pts[i] · pts[j])
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # For each point on the sphere, the distance to the nearest site.
    # We approximate the covering radius by considering the points themselves
    # and the midpoints/arcs between them.
    
    # A better approximation: use the convex hull method.
    # But for speed during optimization, use a simpler metric.
    
    # The covering radius is at least half the maximum of the minimum
    # pairwise distances (for the "worst" gap).
    
    # Let's use: for each point, the distance to its nearest neighbor.
    # The covering radius is bounded by max_i min_{j!=i} angle(i,j) / 2
    # but this isn't tight.
    
    # Better: use the fact that the covering radius is the max circumradius
    # of Delaunay triangles. We can approximate this by looking at triples.
    
    # For efficiency, let's just use the convex hull method directly.
    cr = _covering_radius(pts)
    
    return cr

def _objective_soft(x, beta=50.0):
    """
    Soft objective: approximates the covering radius using a soft-max.
    """
    pts = _normalize(x.reshape(N, 3))
    
    # Compute pairwise angles
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # For each point, find the distance to the nearest other point
    # The covering radius is related to the maximum of these "gaps"
    
    # A practical soft objective: maximize the minimum distance between
    # any two points, which is a proxy for good spreading.
    # But we want to minimize the covering radius.
    
    # Let's use: the covering radius is approximately the maximum over
    # all pairs of the half-angle between them, but this underestimates.
    
    # Use the convex hull covering radius directly.
    cr = _covering_radius(pts)
    return cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_radius = np.pi
    
    # Generate diverse initial configurations
    def _fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def _random_sphere(n, seed=None):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        return _normalize(pts)
    
    def _icosahedron_based(n):
        # Start from icosahedron vertices and add more points
        # Icosahedron has 12 vertices
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        ico = _normalize(ico)
        
        if n <= 12:
            return ico[:n]
        
        # Add random points and optimize
        extra = _random_sphere(n - 12, seed=123)
        return _normalize(np.vstack([ico, extra]))
    
    def _grid_sphere(n):
        # Simple grid-based initialization
        pts = []
        # Use a spherical Fibonacci-like distribution with perturbation
        pts = _fibonacci_sphere(n, offset=0.0)
        return pts
    
    # Create multiple initial configurations
    inits = []
    inits.append(_fibonacci_sphere(N, 0.5))
    inits.append(_fibonacci_sphere(N, 0.0))
    inits.append(_random_sphere(N, seed=1))
    inits.append(_random_sphere(N, seed=42))
    inits.append(_random_sphere(N, seed=123))
    inits.append(_icosahedron_based(N))
    inits.append(_grid_sphere(N))
    
    # Evaluate initial configurations
    for init in inits:
        init = _normalize(init)
        cr = _covering_radius(init)
        if cr < best_radius:
            best_radius = cr
            best_pts = init.copy()
    
    # Multi-stage L-BFGS optimization with increasing beta (if using soft obj)
    # Here we directly minimize the covering radius via L-BFGS on the flat representation
    
    beta_stages = [15, 50, 150, 300, 500]
    
    for stage, beta in enumerate(beta_stages):
        if time.time() > deadline - 10.0:
            break
        
        # Try optimizing from the best current solution
        x0 = best_pts.flatten()
        
        def obj(x):
            pts = _normalize(x.reshape(N, 3))
            return _covering_radius(pts)
        
        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_radius:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_cr = _covering_radius(new_pts)
                if new_cr < best_radius:
                    best_radius = new_cr
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    if time.time() < deadline - 20.0:
        for seed in [7, 99, 200, 500, 1000]:
            if time.time() > deadline - 15.0:
                break
            
            x0 = _random_sphere(N, seed=seed).flatten()
            
            def obj(x):
                pts = _normalize(x.reshape(N, 3))
                return _covering_radius(pts)
            
            try:
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if result.fun < best_radius:
                    new_pts = _normalize(result.x.reshape(N, 3))
                    new_cr = _covering_radius(new_pts)
                    if new_cr < best_radius:
                        best_radius = new_cr
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if best_pts is not None and time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        
        def obj(x):
            pts = _normalize(x.reshape(N, 3))
            return _covering_radius(pts)
        
        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_radius:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_cr = _covering_radius(new_pts)
                if new_cr < best_radius:
                    best_radius = new_cr
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, 0.5)
    
    best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
