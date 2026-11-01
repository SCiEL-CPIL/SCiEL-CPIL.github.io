"""Optimized sphere covering with 50 points using multi-start local optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s limit

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius in radians using convex hull facets."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        # For each facet, find the circumradius
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the points defining this facet
            facet_pts = pts[simplex]
            # Compute circumcenter of the facet
            # For a triangle on the unit sphere, the circumradius is the angular distance
            # from the sphere center to the farthest point in the spherical triangle
            # We use the formula: for points on unit sphere, the circumradius of a facet
            # is the angle from origin to the circumcenter projected onto the sphere
            
            # Method: find the point on the sphere farthest from all facet points
            # that lies in the spherical polygon defined by the facet
            # Simpler: compute the circumradius directly
            
            # For a spherical triangle with side lengths a, b, c,
            # the circumradius R satisfies: sin(R) = sin(a)sin(b)sin(c) / (4*sin(s-a)*sin(s-b)*sin(s-c))
            # where s = (a+b+c)/2
            # But this can be unstable. Alternative: use the fact that for points on unit sphere,
            # the circumradius is related to the distance from origin to the plane of the facet.
            
            # The plane through the facet points has normal n and distance d from origin.
            # The circumradius on the sphere is arccos(d) if d > 0 (the cap radius).
            # Actually, the covering radius contribution from a facet is the maximum angular
            # distance from any point on the sphere to the nearest of the facet's vertices,
            # but constrained to the spherical polygon.
            
            # Better approach: the circumradius of the facet (as a spherical polygon) is
            # the angular radius of the smallest spherical cap containing the facet vertices.
            # This equals pi - arccos(d) where d is the distance from origin to the facet plane,
            # if the facet is "small" (less than a hemisphere).
            
            # Compute the plane through the facet points
            if len(facet_pts) == 3:
                # Triangle
                a = facet_pts[0]
                b = facet_pts[1]
                c = facet_pts[2]
                # Normal vector
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                # Distance from origin to plane
                d = np.dot(n, a)
                # Ensure normal points outward (away from origin)
                if d < 0:
                    d = -d
                    n = -n
                # The circumradius is the angle from the origin to the circumcenter
                # For a spherical triangle, the circumradius R satisfies cos(R) = d
                # Wait, let me think again.
                # The facet is a spherical triangle. The circumradius is the angular distance
                # from the circumcenter to each vertex.
                # The circumcenter lies along the normal direction from the origin.
                # If the plane has equation n·x = d, then the point on the sphere along n
                # is n/||n|| = n (since n is unit). The angular distance from this point
                # to a vertex v is arccos(n·v).
                # For the circumcenter to be equidistant from all three vertices, we need
                # n·a = n·b = n·c = d. So the circumradius is arccos(d).
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                max_radius = max(max_radius, radius)
            elif len(facet_pts) == 4:
                # Tetrahedron facet (shouldn't happen on sphere, but handle gracefully)
                # Use the first 3 points
                a = facet_pts[0]
                b = facet_pts[1]
                c = facet_pts[2]
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                    n = -n
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                max_radius = max(max_radius, radius)
        return max_radius
    except Exception:
        # Fallback: brute force sampling
        theta = np.linspace(0, 2*np.pi, 360)
        phi = np.linspace(0, np.pi, 180)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        # Compute distance from each sample to nearest point
        diffs = samples[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dots = np.sum(samples[:, np.newaxis, :] * pts[np.newaxis, :, :], axis=2)
        min_dots = np.max(dots, axis=1)  # max dot = min angle
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle)

def _objective_and_grad(pts_flat):
    """Objective: covering radius (in radians), with gradient."""
    pts = _normalize(pts_flat.reshape(N, 3))
    
    # Compute covering radius
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            if len(facet_pts) == 3:
                a, b, c = facet_pts
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                    n = -n
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                if radius > max_radius:
                    max_radius = radius
                    worst_facet = simplex
            elif len(facet_pts) == 4:
                a, b, c = facet_pts[:3]
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.dot(n, a)
                if d < 0:
                    d = -d
                    n = -n
                radius = np.arccos(np.clip(d, -1.0, 1.0))
                if radius > max_radius:
                    max_radius = radius
                    worst_facet = simplex
        
        if worst_facet is None:
            return max_radius, np.zeros_like(pts_flat)
        
        # Gradient: for the worst facet, the gradient of arccos(d) with respect to points
        # d = n·a where n is the unit normal
        # This is complex; use finite differences or a simpler gradient
        # For now, return zero gradient and let L-BFGS use numerical differentiation
        return max_radius, None
    except Exception:
        # Fallback
        theta = np.linspace(0, 2*np.pi, 180)
        phi = np.linspace(0, np.pi, 90)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dots = np.sum(samples[:, np.newaxis, :] * pts[np.newaxis, :, :], axis=2)
        min_dots = np.max(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle), None

def _local_optimize(pts0, time_limit=10.0):
    """Run local optimization from initial points."""
    start_time = time.time()
    pts = _normalize(pts0.copy())
    
    def obj(x):
        pts_local = _normalize(x.reshape(N, 3))
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts_local)
            max_r = 0.0
            for simplex in hull.simplices:
                fp = pts_local[simplex]
                if len(fp) == 3:
                    a, b, c = fp
                    n = np.cross(b - a, c - a)
                    nn = np.linalg.norm(n)
                    if nn < 1e-12:
                        continue
                    n = n / nn
                    d = np.dot(n, a)
                    if d < 0:
                        d = -d
                    r = np.arccos(np.clip(d, -1.0, 1.0))
                    max_r = max(max_r, r)
                elif len(fp) == 4:
                    a, b, c = fp[:3]
                    n = np.cross(b - a, c - a)
                    nn = np.linalg.norm(n)
                    if nn < 1e-12:
                        continue
                    n = n / nn
                    d = np.dot(n, a)
                    if d < 0:
                        d = -d
                    r = np.arccos(np.clip(d, -1.0, 1.0))
                    max_r = max(max_r, r)
            return max_r
        except Exception:
            # Fallback
            theta = np.linspace(0, 2*np.pi, 120)
            phi = np.linspace(0, np.pi, 60)
            TH, PH = np.meshgrid(theta, phi)
            X = np.sin(PH) * np.cos(TH)
            Y = np.sin(PH) * np.sin(TH)
            Z = np.cos(PH)
            samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            dots = np.sum(samples[:, np.newaxis, :] * pts_local[np.newaxis, :, :], axis=2)
            min_dots = np.max(dots, axis=1)
            return np.max(np.arccos(np.clip(min_dots, -1.0, 1.0)))
    
    result = minimize(obj, pts.ravel(), method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.success or result.fun < _covering_radius(pts):
        pts = _normalize(result.x.reshape(N, 3))
    
    return pts

def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    
    # Multi-start local optimization
    best_pts = None
    best_radius = np.inf
    
    # Start 1: Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    r0 = _covering_radius(pts0)
    if best_radius > r0:
        best_radius = r0
        best_pts = pts0.copy()
    
    # Try local optimization from Fibonacci
    if time.time() - start_time < DEADLINE - 15:
        pts_opt = _local_optimize(pts0, time_limit=15.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
    
    # Start 2: Random with seed 0
    if time.time() - start_time < DEADLINE - 15:
        pts0 = _random_sphere(N, seed=0)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _local_optimize(pts0, time_limit=10.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
    
    # Start 3: Random with seed 1
    if time.time() - start_time < DEADLINE - 10:
        pts0 = _random_sphere(N, seed=1)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _local_optimize(pts0, time_limit=10.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
    
    # Start 4: Random with seed 2
    if time.time() - start_time < DEADLINE - 10:
        pts0 = _random_sphere(N, seed=2)
        r0 = _covering_radius(pts0)
        if r0 < best_radius:
            best_radius = r0
            best_pts = pts0.copy()
        pts_opt = _local_optimize(pts0, time_limit=10.0)
        r_opt = _covering_radius(pts_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_pts = pts_opt.copy()
    
    # Final polish: one more local optimization from best
    if time.time() - start_time < DEADLINE - 5:
        pts_final = _local_optimize(best_pts, time_limit=5.0)
        r_final = _covering_radius(pts_final)
        if r_final < best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    return best_pts
