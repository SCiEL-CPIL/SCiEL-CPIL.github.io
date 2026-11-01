"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(v):
    """Normalize vectors along the last axis."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in degrees) for a set of points on the unit sphere.
    The covering radius is the maximum circumradius of the facets of the convex hull
    that have an empty cap (i.e., the circumcenter of each facet is in the direction
    away from the origin, meaning the cap is empty of other points).
    """
    pts = np.asarray(pts)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a dense grid to estimate covering radius
        return _covering_radius_grid(pts)
    
    # For each facet (simplex), compute the circumradius
    max_cr = 0.0
    for simplex in hull.simplices:
        # simplex is a set of indices into pts
        tri = pts[simplex]  # shape (d+1, 3) where d=2 for sphere -> triangles
        
        # Compute the circumcenter of the triangle on the sphere
        # For a triangle with vertices a, b, c on the unit sphere,
        # the circumcenter is the normalized vector: a + b + c adjusted,
        # but more precisely we need the point equidistant from all three.
        
        # The circumradius of a spherical triangle is the angular distance from
        # the circumcenter to any vertex. For small triangles, this is approximately
        # the planar circumradius.
        
        # Exact spherical circumcenter:
        # The circumcenter is along the direction of the cross product of (b-a) and (c-a)
        # but that gives the normal. Actually, for a triangle on the unit sphere,
        # the circumcenter is the normalized sum of the vertices if the triangle is small,
        # but more generally we solve for a point x on the sphere equidistant from a, b, c.
        
        # A robust method: the circumcenter of triangle (a,b,c) on the unit sphere
        # is in the direction of: a*(sin(2*angle_b) * ...) 
        # Simpler: use the formula that the circumcenter direction is proportional to:
        # a * (|b|^2 |c|^2 - ...) -- this gets complex.
        
        # Practical approach: the circumcenter is the normalized vector
        # w = a + b + c + k * (a x b x c) ... no.
        
        # Let's use: the circumcenter is the point on the sphere equidistant from a, b, c.
        # This means |x-a|^2 = |x-b|^2 = |x-c|^2
        # |x|^2 - 2 x.a + |a|^2 = |x|^2 - 2 x.b + |b|^2
        # Since |x|=|a|=|b|=|c|=1:
        # x.a = x.b = x.c
        # So x is in the null space of [a-b, a-c] and |x|=1.
        
        a, b, c = tri[0], tri[1], tri[2]
        
        # Solve x . (a-b) = 0 and x . (a-c) = 0
        # x is perpendicular to (a-b) and (a-c)
        # So x is parallel to (a-b) x (a-c)
        v1 = a - b
        v2 = a - c
        x_dir = np.cross(v1, v2)
        x_norm = np.linalg.norm(x_dir)
        
        if x_norm < 1e-12:
            # Degenerate triangle
            continue
        
        x = x_dir / x_norm
        
        # Check if the cap is empty: the circumcenter should be on the "outside"
        # i.e., the dot product with the facet normal should be positive
        # The facet normal (pointing outward) is in the direction of a+b+c (approximately)
        # More precisely, for the convex hull, the outward normal of the facet is
        # in the direction such that all other points are on the inside.
        
        # For the covering radius, we want the circumradius of the facet.
        # The angular distance from x to a is arccos(x . a)
        cr = np.arccos(np.clip(x.dot(a), -1.0, 1.0))
        
        # But we need to check if this is the correct circumcenter (the one with empty cap)
        # The cap is empty if no other point is closer to x than the vertices.
        # For the convex hull facets, by construction, the cap in the direction of x
        # contains no other points if x is the outward circumcenter.
        
        # Actually, for the covering radius, we want the largest circumradius among
        # all facets. The circumcenter we found might be the "inner" one.
        # The two candidates are ±x. We want the one where the cap is empty.
        # For a convex hull facet, the outward normal direction has the property
        # that all other points are on the opposite side.
        
        # Let's check both ±x and pick the one that is farther from the origin's opposite
        # i.e., the one where the dot product with the centroid of the triangle is positive
        centroid = (a + b + c) / 3.0
        centroid = _normalize(centroid)[0]
        
        if x.dot(centroid) < 0:
            x = -x
        
        cr = np.arccos(np.clip(x.dot(a), -1.0, 1.0))
        
        if cr > max_cr:
            max_cr = cr
    
    return np.degrees(max_cr)


def _covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Fallback covering radius estimation using a dense grid."""
    pts = _normalize(np.asarray(pts))
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find the minimum angular distance to any config point
    # Use chunking to avoid memory issues
    max_min_dist = 0.0
    chunk_size = 10000
    for i in range(0, len(grid), chunk_size):
        chunk = grid[i:i+chunk_size]
        # Compute dot products
        dots = chunk @ pts.T  # shape (chunk_size, N)
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distance
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        chunk_max = np.max(min_dists)
        if chunk_max > max_min_dist:
            max_min_dist = chunk_max
    
    return np.degrees(max_min_dist)


def _gradient_descent(pts, max_iters=5000, tol=1e-10, time_limit=None, start_time=None):
    """
    Gradient descent to minimize the covering radius.
    Uses the fact that the covering radius is determined by the largest facet circumradius.
    We move points to reduce the largest circumradii.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = _normalize(np.array(pts, dtype=float))
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    lr = 0.01
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute current covering radius and identify the critical facet
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
        
        # Compute gradients: for each point, compute the gradient of the covering radius
        # with respect to that point's position on the sphere.
        # This is complex, so we use a simpler approach:
        # Move points away from the circumcenter of the largest facet.
        
        # Find the largest facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_cr = 0.0
        max_facet = None
        max_circumcenter = None
        
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            v1 = a - b
            v2 = a - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-12:
                continue
            x = x_dir / x_norm
            
            centroid = _normalize((a + b + c) / 3.0)[0]
            if x.dot(centroid) < 0:
                x = -x
            
            cr_facet = np.arccos(np.clip(x.dot(a), -1.0, 1.0))
            if cr_facet > max_cr:
                max_cr = cr_facet
                max_facet = simplex
                max_circumcenter = x
        
        if max_facet is None:
            break
        
        # Move the vertices of the largest facet away from the circumcenter
        # This reduces the circumradius
        for idx in max_facet:
            p = pts[idx]
            # Move away from circumcenter: gradient direction is -circumcenter
            # But we need to stay on the sphere
            # The tangent direction away from circumcenter is:
            # proj_tangent(-circumcenter) = -circumcenter + (circumcenter . p) * p
            grad = -max_circumcenter + (max_circumcenter.dot(p)) * p
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad /= grad_norm
                pts[idx] = pts[idx] + lr * grad
                pts[idx] = _normalize(pts[idx])
        
        # Also add a small repulsion between all points to avoid coincidences
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.1:  # Only repel when close
                    force = diff / (dist ** 2 + 1e-6)
                    pts[i] += lr * 0.1 * force
                    pts[j] -= lr * 0.1 * force
        
        pts = _normalize(pts)
        
        # Check convergence
        if it > 0 and abs(cr - best_cr) < tol:
            if cr >= best_cr:
                break
    
    return best_pts, best_cr


def _polish(pts, max_iters=1000, time_limit=None, start_time=None):
    """
    High-precision polish: alternate between reducing the largest facet circumradius
    and maintaining good separation.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = _normalize(np.array(pts, dtype=float))
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    lr = 0.005
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        cr = _covering_radius(pts)
        if cr < best_cr - 1e-12:
            best_cr = cr
            best_pts = pts.copy()
        
        # Find the largest facet and its circumcenter
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_cr = 0.0
        max_facet = None
        max_circumcenter = None
        
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            v1 = a - b
            v2 = a - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-12:
                continue
            x = x_dir / x_norm
            
            centroid = _normalize((a + b + c) / 3.0)[0]
            if x.dot(centroid) < 0:
                x = -x
            
            cr_facet = np.arccos(np.clip(x.dot(a), -1.0, 1.0))
            if cr_facet > max_cr:
                max_cr = cr_facet
                max_facet = simplex
                max_circumcenter = x
        
        if max_facet is None:
            break
        
        # Move vertices of the largest facet away from circumcenter
        for idx in max_facet:
            p = pts[idx]
            grad = -max_circumcenter + (max_circumcenter.dot(p)) * p
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad /= grad_norm
                pts[idx] = pts[idx] + lr * grad
                pts[idx] = _normalize(pts[idx])
        
        # Global repulsion to maintain separation
        for i in range(N):
            repulse = np.zeros(3)
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.15:
                    force = diff / (dist ** 3 + 1e-6)
                    repulse += force
            if np.linalg.norm(repulse) > 1e-12:
                # Project to tangent plane
                repulse = repulse - (repulse.dot(pts[i])) * pts[i]
                pts[i] = pts[i] + lr * 0.05 * repulse
                pts[i] = _normalize(pts[i])
        
        pts = _normalize(pts)
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Safety margin before 150s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Gradient descent
    remaining = total_time_limit - (time.time() - start_time) - 20.0  # Reserve 20s for polish
    if remaining > 10.0:
        pts, cr = _gradient_descent(pts, max_iters=3000, time_limit=remaining, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 2: Polish
    remaining = total_time_limit - (time.time() - start_time) - 5.0
    if remaining > 5.0:
        pts, cr = _polish(best_pts, max_iters=2000, time_limit=remaining, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
