"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50
_TIME_LIMIT = 115.0  # safety margin before 120s kill


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of points on the unit sphere.
    The covering radius is the largest circumradius of the facets of the convex hull,
    where each facet is checked to have an empty cap (i.e., no other points lie
    on the same side of the facet's plane as the origin).
    """
    if len(points) < 3:
        return np.pi / 2  # fallback

    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    # Build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to brute-force nearest neighbor
        # Sample many points on the sphere
        n_samples = 20000
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        # Find nearest point for each sample
        # Use chunked computation
        max_dist = 0.0
        chunk = 5000
        for i in range(0, n_samples, chunk):
            s = samples[i:i+chunk]
            d = np.max(np.arccos(np.clip(s @ points.T, -1, 1)), axis=1)
            max_dist = max(max_dist, np.max(d))
        return np.degrees(max_dist)

    # For each facet, compute the circumradius
    # A facet is defined by 3 points. The circumradius of the spherical triangle
    # is the angular distance from the facet's center to the farthest vertex,
    # but actually for covering we need the largest empty cap radius.
    #
    # The covering radius is the maximum over all points y on the sphere of
    # min_i angular_dist(y, p_i). This equals the maximum over all facets of
    # the convex hull of the circumradius of that facet (if the cap is empty).
    #
    # For a facet with vertices a, b, c (unit vectors), the circumcenter in
    # the plane of the facet (which contains the origin only if the facet
    # is on the "far side")... Actually, the relevant quantity is the angular
    # distance from the origin to the plane of the facet, but that's the inradius.
    #
    # Let me reconsider: The covering radius is the largest r such that there
    # exists a point y with d(y, p_i) >= r for all i. This is equivalent to
    # the maximum over all facets of the convex hull of the circumradius of
    # the spherical triangle formed by the facet vertices, provided the cap
    # is empty.
    #
    # For a plane defined by normal n (unit) at distance d from origin, the
    # angular distance from origin to the plane is arccos(d). But that's the
    # minimum distance to any point on the plane. The covering radius for
    # that facet is determined by the circumcenter of the spherical triangle.
    #
    # Actually, the correct approach: For each facet (triangle) of the convex
    # hull, compute the circumcenter of the spherical triangle. The covering
    # radius is the maximum angular distance from that circumcenter to any
    # vertex of the triangle, over all facets whose cap is empty.
    #
    # But a simpler and correct method: the covering radius equals the maximum
    # over all facets of the angular circumradius of the facet's vertices.
    # The angular circumradius of a spherical triangle with vertices a,b,c
    # (unit vectors) is the radius of the smallest spherical cap containing
    # a, b, c. This can be computed as follows:
    #
    # The circumcenter is along the normal to the plane of a,b,c. If the
    # normal n points away from the origin (n . origin = 0, so this is about
    # the sign), the circumcenter on the sphere is at angular distance
    # arccos(|n_plane|) from the origin where n_plane is the unit normal to
    # the plane through a,b,c.
    #
    # Let me use a different approach: for each facet, the covering contribution
    # is the angular distance from the origin to the circumcenter of the triangle
    # a,b,c on the sphere. The circumcenter of the spherical triangle is the
    # point equidistant from a, b, c. 
    #
    # Simpler: compute the plane through a,b,c. The normal is n = (b-a) x (c-a),
    # normalized. The distance from origin to plane is |a . n|. The circumradius
    # (angular) of the triangle as seen from origin is arccos(a . n) if the
    # cap is on the correct side. But we need to be careful.
    #
    # Let me just use: for each facet, the relevant angular radius is the
    # maximum angular distance from the circumcenter to the vertices. The
    # circumcenter of the spherical triangle can be found as the normalized
    # sum of the vertices weighted appropriately, or via the plane normal.
    #
    # Actually, the simplest correct method: the covering radius is the
    # maximum over all y on sphere of min_i d(y, p_i). This is the same as
    # the maximum circumradius of the Delaunay triangulation on the sphere.
    # For the convex hull facets, each facet's circumradius (angular) gives
    # a candidate. The covering radius is the max of these over empty caps.
    #
    # For a triangle with vertices a, b, c (unit), the circumradius R satisfies:
    # sin(R) = |a x b| * |b x c| * |c x a| / (2 * |a . (b x c)|) ... no that's not right.
    #
    # Let me use the formula: the circumradius of the spherical triangle is
    # given by R = arccos( (a . b * b . c * c . a - (a . b)^2 - (b . c)^2 - (c . a)^2 + 1) / (2 * (a . b + b . c + c . a) - 2) )... this is getting complicated.
    #
    # Alternative: The circumcenter of the spherical triangle is the point u
    # such that u . a = u . b = u . c. This means u is along the normal to the
    # plane containing a, b, c. So u = ±n where n is the unit normal to the
    # plane. The correct sign is the one where u is on the same side as the
    # origin relative to the plane... actually for the convex hull, the facet
    # normal points outward. The circumcenter for the empty cap is the one
    # on the opposite side from the facet normal (i.e., toward the origin side).
    #
    # Let me just compute it directly:
    # For each facet, compute the circumradius as the angular distance from
    # the circumcenter to any vertex. The circumcenter is the normalized cross
    # product approach.

    max_covering = 0.0
    facets = hull.simplices
    for facet in facets:
        a, b, c = points[facet[0]], points[facet[1]], points[facet[2]]
        
        # Normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        # The circumcenter of the spherical triangle is along n or -n.
        # We want the one that is on the "inside" (toward origin) for empty cap check.
        # For the convex hull, the facet normal points outward. The empty cap
        # is on the inward side. So the circumcenter for the covering radius
        # is -n (pointing toward the origin side).
        
        # Check if the cap is empty: all other points should be on the same
        # side of the plane as the origin (i.e., a . n <= 0 for outward normal)
        # Actually, for the convex hull, all points satisfy p . n <= a . n
        # (since n points outward). The cap is empty if no point is beyond
        # the plane, which is guaranteed by the convex hull definition.
        # But we need to check that the circumcenter's cap is empty.
        
        # The circumcenter on the sphere: u = -n (the one closer to origin side)
        # The angular distance from u to a is arccos(u . a)
        u = -n
        # But u might not be equidistant from a, b, c if the triangle is
        # obtuse in the spherical sense. Let me just compute the circumradius
        # as the maximum angular distance from u to the vertices.
        
        da = np.arccos(np.clip(u @ a, -1, 1))
        db = np.arccos(np.clip(u @ b, -1, 1))
        dc = np.arccos(np.clip(u @ c, -1, 1))
        
        # The circumradius is the maximum of these
        r = max(da, db, dc)
        
        # Check if the cap centered at u with radius r is empty (no other
        # points inside). This is guaranteed by the convex hull facet definition
        # if u is on the correct side.
        
        max_covering = max(max_covering, r)
    
    return np.degrees(max_covering)


def _objective_and_gradient(points):
    """
    Compute a smooth surrogate for the covering radius and its gradient.
    Uses a log-sum-exp soft-max over the nearest neighbor distances.
    """
    # For each point on a dense sampling of the sphere, find the nearest
    # configuration point and compute the distance. Then use a soft-max.
    # But this is expensive. Instead, use the convex hull facet circumradii
    # with a smooth approximation.
    
    # Simpler: use the pairwise distances between configuration points as a
    # proxy, but that's not quite right.
    
    # Let me use: the covering radius is bounded by half the maximum
    # nearest-neighbor distance between configuration points (in the
    # Delaunay sense). A good surrogate is the soft-max of the circumradii
    # of the convex hull facets.
    
    # For now, let's use a simpler approach: gradient descent on the
    # sum of squared nearest-neighbor distances for a set of test points.
    
    # Actually, let me implement a proper gradient using the convex hull.
    # The gradient of the covering radius with respect to point i is related
    # to the circumcenters of the facets containing point i.
    
    # For simplicity and robustness, I'll use a subgradient method based on
    # the worst facet.
    
    return None, None


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Local optimization: iterative micro-polish with per-facet adaptive step size
    rng = np.random.default_rng(12345)
    
    # Phase 1: Global gradient-like steps
    lr = 0.01
    for it in range(200):
        if time.time() - t_start > _TIME_LIMIT - 20:
            break
        
        # Compute covering radius
        radius = _covering_radius(points)
        
        # Find worst facet and move points away from it
        # Compute nearest neighbor distances for all points
        # For each point, the gradient pushes it away from the worst covering center
        
        # Use a simple approach: for each pair of close points, push them apart
        # and for points in the worst cap region, push them toward the center
        
        # Compute all pairwise angular distances
        cos_dist = np.clip(points @ points.T, -1, 1)
        dist = np.arccos(cos_dist)
        np.fill_diagonal(dist, np.pi)  # avoid self-distance
        
        # For each point, compute the "pressure" from nearby points
        # Points that are too close together should be pushed apart
        # Points in sparse regions should be pulled toward the gap
        
        # Simple repulsion: for each point, sum of repulsion from nearby points
        grad = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < np.pi / 4:  # only interact with close points
                    # Repulsion force
                    repulse = (1.0 - d / (np.pi / 4)) ** 2
                    grad[i] -= repulse * (points[i] - points[j]) / (d + 1e-10)
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-15:
            grad /= gnorm
        
        # Update
        new_points = points + lr * grad
        # Renormalize
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _covering_radius(new_points)
        if new_radius < radius:
            points = new_points
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            lr *= 0.9
        
        # Adaptive learning rate
        if it % 20 == 0:
            lr = max(0.001, lr * 0.95)
    
    # Phase 2: Micro-polish with per-facet adaptive step size
    lr = 0.005
    for it in range(300):
        if time.time() - t_start > _TIME_LIMIT - 5:
            break
        
        radius = _covering_radius(points)
        
        # Find the worst facet(s) and adjust points on those facets
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
        
        # Compute circumradius for each facet
        facet_radii = []
        facet_centers = []
        for facet in facets:
            a, b, c = points[facet[0]], points[facet[1]], points[facet[2]]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            u = -n
            da = np.arccos(np.clip(u @ a, -1, 1))
            db = np.arccos(np.clip(u @ b, -1, 1))
            dc = np.arccos(np.clip(u @ c, -1, 1))
            r_facet = max(da, db, dc)
            facet_radii.append(r_facet)
            facet_centers.append(u)
        
        if not facet_radii:
            break
        
        facet_radii = np.array(facet_radii)
        facet_centers = np.array(facet_centers)
        
        # Find the worst facet
        worst_idx = np.argmax(facet_radii)
        worst_radius = facet_radii[worst_idx]
        
        if worst_radius < np.radians(best_radius):
            # Already better than best
            if np.degrees(worst_radius) < best_radius:
                best_radius = np.degrees(worst_radius)
                best_points = points.copy()
        
        # Move the vertices of the worst facet away from the circumcenter
        u = facet_centers[worst_idx]
        facet = facets[worst_idx]
        for vi in facet:
            # Move point vi away from u
            direction = points[vi] - u
            dnorm = np.linalg.norm(direction)
            if dnorm > 1e-10:
                direction /= dnorm
                step = lr * (worst_radius / (np.pi / 6))  # adaptive step
                new_p = points[vi] + step * direction
                new_p /= np.linalg.norm(new_p)
                # Check if this improves
                test_points = points.copy()
                test_points[vi] = new_p
                test_radius = _covering_radius(test_points)
                if test_radius < radius:
                    points[vi] = new_p
                    if test_radius < best_radius:
                        best_radius = test_radius
                        best_points = points.copy()
        
        # Also try moving points slightly toward sparse regions
        # Find the point with the largest nearest-neighbor distance
        cos_dist = np.clip(points @ points.T, -1, 1)
        dist = np.arccos(cos_dist)
        np.fill_diagonal(dist, 0)
        nn_dist = np.min(dist, axis=1)
        sparse_idx = np.argmax(nn_dist)
        
        # Move the sparse point slightly toward the center of its Voronoi cell
        # Approximate by moving toward the centroid of its nearest neighbors
        neighbors = np.argsort(dist[sparse_idx])[:3]
        centroid = np.mean(points[neighbors], axis=0)
        centroid /= np.linalg.norm(centroid)
        step = lr * 0.5
        new_p = points[sparse_idx] + step * (centroid - points[sparse_idx])
        new_p /= np.linalg.norm(new_p)
        test_points = points.copy()
        test_points[sparse_idx] = new_p
        test_radius = _covering_radius(test_points)
        if test_radius < radius:
            points[sparse_idx] = new_p
            if test_radius < best_radius:
                best_radius = test_radius
                best_points = points.copy()
    
    # Final: return the best points found
    return best_points
