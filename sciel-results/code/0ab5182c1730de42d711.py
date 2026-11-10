"""Rewritten program: spherical covering with 50 caps using force-based optimization."""
import numpy as np
import time

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians by finding the largest circumradius of hull facets."""
    from scipy.spatial import ConvexHull
    from scipy.spatial import Delaunay
    
    pts = _normalize(pts)
    # Use Delaunay triangulation on the sphere via convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each facet (triangle), compute the circumradius on the sphere
    # The covering radius is the maximum over all facets of the angular distance
    # from the facet's center (in the spherical sense) to its vertices, but more precisely,
    # it's the maximum over the sphere of the distance to the nearest point.
    # 
    # A better approach: the covering radius equals the maximum, over all points y on the sphere,
    # of the minimum distance to any site. This is achieved at a vertex of the Voronoi diagram
    # on the sphere, which corresponds to circumcenters of Delaunay triangles.
    
    # Get Delaunay triangles from the convex hull facets
    # Each facet is a triangle (for 3D convex hull of points in general position)
    facets = hull.simplices  # shape (n_facets, 3)
    
    max_radius = 0.0
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant (angularly) from a, b, c
        # On the sphere, this is the intersection of the sphere with the plane 
        # perpendicular to the triangle's plane passing through the origin... 
        # Actually, the circumcenter of a spherical triangle is the point on the sphere
        # equidistant from all three vertices. It lies along the normal to the plane 
        # containing a, b, c, but that's only if the triangle is right. 
        # 
        # More carefully: the circumcenter is found by solving for a point p on the unit sphere
        # such that p·a = p·b = p·c. This means p is in the null space of [a-b, a-c] 
        # (in the sense that p is orthogonal to both a-b and a-c). So p is parallel to 
        # the cross product (a-b) × (a-c), which is the normal to the triangle's plane.
        
        n = np.cross(a - b, a - c)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        p = n / n_norm
        
        # Check which side: we want the circumcenter that is on the "far" side (the one 
        # that is the center of the cap covering this facet). Actually, for the covering 
        # problem, we need the point that is the farthest from all sites, which is the 
        # circumcenter of the Delaunay triangle on the opposite side of the triangle 
        # from the origin... 
        # 
        # Actually, the Voronoi vertex (circumcenter) for a Delaunay triangle on the sphere:
        # The two candidate points are ±p. We need the one that is NOT inside the convex 
        # hull of the other points... Actually, for the covering radius, we want the 
        # circumcenter that is the center of the largest empty cap. The correct one is 
        # the one that is on the opposite side of the triangle from the bulk of the points.
        # 
        # Simpler: just compute the angular distance from p (and -p) to the three vertices,
        # and take the one where all three distances are consistent (i.e., the circumcenter
        # should have equal distances to a, b, c).
        
        # p should satisfy p·a ≈ p·b ≈ p·c
        da = np.dot(p, a)
        db = np.dot(p, b)
        dc = np.dot(p, c)
        
        # If the distances are not equal, try -p
        if abs(da - db) > 1e-8 or abs(da - dc) > 1e-8:
            p = -p
            da = np.dot(p, a)
            db = np.dot(p, b)
            dc = np.dot(p, c)
        
        # The angular distance from p to a is arccos(p·a)
        # Clamp for numerical safety
        da = np.clip(da, -1.0, 1.0)
        r = np.arccos(da)
        
        if r > max_radius:
            max_radius = r
    
    # Also check: the covering radius might be determined by a point that is not a 
    # circumcenter but rather on an edge or vertex of the Voronoi diagram. However, 
    # for a proper Delaunay triangulation, the maximum is always at a circumcenter.
    
    return max_radius

def _covering_radius_fast(pts):
    """Faster covering radius estimate using a dense grid (for optimization)."""
    pts = _normalize(pts)
    # Create a dense grid of test points on the sphere
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    
    # Compute distances from each test point to all sites
    # X is (M, 3), pts is (N, 3)
    # dist[i, j] = angle between X[i] and pts[j]
    dots = X @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    min_dists = np.min(dists, axis=1)  # (M,)
    return np.max(min_dists)

def _force_based_step(pts, step_size):
    """One step of force-based optimization: push points away from their Voronoi neighbors."""
    pts = _normalize(pts)
    
    # Compute pairwise dot products
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Force: each point is pushed away from nearby points
    # Use a repulsive force that is stronger for closer points
    # F_ij = (1/d_ij^2) * (p_i - p_j) for i != j, but only for close pairs
    
    forces = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            if d < 0.5:  # Only consider close pairs
                # Repulsive force proportional to 1/d^2, direction away from j
                # In Cartesian: force on i from j is proportional to (p_i - p_j)/d^2
                # But we need to project onto the tangent plane
                force_mag = 1.0 / (d * d + 1e-10)
                direction = pts[i] - pts[j]
                # Project onto tangent plane at pts[i]
                tangential = direction - np.dot(direction, pts[i]) * pts[i]
                forces[i] += force_mag * tangential
    
    # Normalize forces and take a step
    forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces_norm[forces_norm < 1e-12] = 1.0
    forces = forces / forces_norm
    
    new_pts = pts + step_size * forces
    new_pts = _normalize(new_pts)
    
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_pts = None
    best_radius = np.pi
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Evaluate initial
    init_radius = _covering_radius_fast(pts)
    best_radius = init_radius
    best_pts = pts.copy()
    
    # Force-based optimization with annealing
    step_size = 0.05
    n_iterations = 0
    no_improve = 0
    
    while time.time() < deadline:
        # Compute current radius (fast version for speed)
        current_radius = _covering_radius_fast(pts)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Take a force-based step
        new_pts = _force_based_step(pts, step_size)
        
        # Evaluate new configuration
        new_radius = _covering_radius_fast(new_pts)
        
        if new_radius < current_radius:
            pts = new_pts
            # Decrease step size slowly
            step_size *= 0.999
        else:
            # Rejected: increase step size slightly to escape local minima
            step_size *= 1.02
            step_size = min(step_size, 0.1)
        
        n_iterations += 1
        
        # Every 500 iterations, do a more accurate evaluation and possibly refine
        if n_iterations % 500 == 0 and time.time() < deadline - 10:
            acc_radius = _covering_radius(best_pts)
            if acc_radius < best_radius:
                best_radius = acc_radius
    
    # Final polish: use the best configuration found
    final_pts = best_pts
    
    # Verify it's valid
    final_pts = _normalize(final_pts)
    if not np.all(np.isfinite(final_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        final_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        final_pts = _normalize(final_pts)
    
    return final_pts
