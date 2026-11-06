"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    Uses the convex hull: the covering radius is the max circumradius of facets
    whose empty-cap condition holds (i.e., the facet's pole is on the sphere and
    the cap defined by the facet contains no other points)."""
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
    
    covering_radius = 0.0
    n = len(points)
    
    for simplex in hull.simplices:
        # Get the facet points
        facet_pts = points[simplex]
        k = len(simplex)
        
        # Find the circumcenter of the facet (simplex)
        # For a simplex with vertices v0, v1, ..., v_{k-1}, the circumcenter
        # satisfies |c - vi|^2 = R^2 for all i.
        # We solve: 2 * vi . c = |vi|^2 + R^2 - |c|^2
        # Since all vi are on the unit sphere, |vi|^2 = 1.
        # Let h = 1 - |c|^2 (which is related to the inradius/circumradius)
        # Actually, for points on unit sphere, the circumcenter of the facet
        # lies on the line through the origin and the centroid of the facet (for symmetric cases)
        # More generally, we solve the linear system.
        
        # The circumcenter c satisfies:
        # |c|^2 - 2*vi.c + |vi|^2 = R^2
        # |c|^2 - 2*vi.c + 1 = R^2
        # So 2*vi.c = |c|^2 + 1 - R^2 for all i
        # Let s = |c|^2 + 1 - R^2, then vi.c = s/2 for all i.
        # This means c is in the span of the facet points, and the projection
        # of c onto each vi is the same.
        
        # Alternative approach: the circumcenter of a simplex on the unit sphere
        # can be found by solving a linear system.
        # Let A = [v1 - v0, v2 - v0, ..., v_{k-1} - v0]^T (k-1 x 3)
        # The circumcenter c satisfies:
        # |c - v0|^2 = |c - vi|^2 => 2*ci.v0 - 2*ci.vi = |v0|^2 - |vi|^2 = 0
        # So ci.(vi - v0) = 0 for i=1..k-1
        # And c is in the affine hull: c = v0 + A^T @ t for some t
        
        # Simpler: use the fact that for a simplex on the unit sphere, the 
        # circumcenter is proportional to the solution of a linear system.
        
        # Let's use the formula: the circumcenter of points v0,...,v_{k-1} on unit sphere
        # is given by solving:
        # M @ c = b, where M[i] = vi - v0, b[i] = 0 (since |vi|^2 = |v0|^2 = 1)
        # This gives c in the direction that is orthogonal to all edge vectors from v0.
        # But we also need c to be in the affine hull.
        
        # Actually, the circumcenter lies in the affine hull of the simplex.
        # c = v0 + sum_{i=1}^{k-1} ti * (vi - v0)
        # And |c - vj|^2 = |c - v0|^2 for all j.
        
        # Let's just compute it numerically.
        # The circumradius R and circumcenter c satisfy:
        # For each vertex vi: |c - vi|^2 = R^2
        # Expanding: |c|^2 - 2 c.vi + 1 = R^2
        # So c.vi = (|c|^2 + 1 - R^2)/2 =: alpha for all i.
        
        # This means c is a vector such that its dot product with each vi is the same.
        # c = alpha * sum_i wi * vi where the weights are chosen so that c.vi = alpha.
        
        # Linear system: Let V be the matrix with rows vi (k x 3).
        # We want c such that V @ c = alpha * ones(k).
        # c = V^T @ (V V^T)^{-1} @ alpha * ones(k) ... but this is for the case where c is in the span.
        
        # Since c must be in the affine hull of the simplex, and the simplex is k-dimensional
        # (k = simplex dimension + 1), c is determined uniquely.
        
        # Let's use a direct approach:
        # c = v0 + A^T @ t, where A has rows (vi - v0) for i=1..k-1
        # |c - v0|^2 = |c - vi|^2
        # |A^T t|^2 = |A^T t - (vi - v0)|^2
        # 0 = -2 t^T A (vi - v0) + |vi - v0|^2
        # A A^T t = (1/2) A @ |vi - v0|^2
        
        if k == 1:
            # Single point, circumradius is 0
            continue
        elif k == 2:
            # Two points: circumcenter is midpoint direction
            v0, v1 = facet_pts[0], facet_pts[1]
            d = v1 - v0
            # Midpoint
            mid = (v0 + v1) / 2.0
            # The circumcenter is along the direction of mid, scaled so that it's equidistant
            # |c - v0|^2 = |c - v1|^2 => c is on the perpendicular bisector plane
            # For two points on unit sphere, the circumcenter is mid / |mid| * R where R is chosen
            # Actually, the circumcenter c satisfies c.v0 = c.v1, which is satisfied by any c perpendicular to d.
            # The one in the affine hull is c = mid.
            c = mid
            R = np.linalg.norm(c - v0)
        elif k == 3:
            v0, v1, v2 = facet_pts
            A = np.array([v1 - v0, v2 - v0])
            b = np.array([np.dot(v1 - v0, v1 - v0), np.dot(v2 - v0, v2 - v0)]) / 2.0
            try:
                t = np.linalg.solve(A @ A.T, b)
            except np.linalg.LinAlgError:
                continue
            c = v0 + A.T @ t
            R = np.linalg.norm(c - v0)
        elif k == 4:
            v0, v1, v2, v3 = facet_pts
            A = np.array([v1 - v0, v2 - v0, v3 - v0])
            b = np.array([np.dot(v1 - v0, v1 - v0), np.dot(v2 - v0, v2 - v0), np.dot(v3 - v0, v3 - v0)]) / 2.0
            try:
                t = np.linalg.solve(A @ A.T, b)
            except np.linalg.LinAlgError:
                continue
            c = v0 + A.T @ t
            R = np.linalg.norm(c - v0)
        else:
            continue
        
        # The covering radius contribution from this facet is the angular distance
        # from the facet's circumcenter direction to the nearest point.
        # The cap defined by the facet: points y on the sphere such that y.c >= |c|
        # (assuming c points outward). The angular radius of this cap is arccos(|c|/|c|) ... 
        # Wait. The facet defines a plane. The cap is the set of points on the sphere
        # on the same side of the plane as the origin (for a convex hull containing the origin).
        # The angular distance from the circumcenter of the facet (projected to sphere) to any vertex
        # is the circumradius.
        
        # The covering radius is the maximum over all facets of the circumradius,
        # but only if the cap is empty (no other points inside).
        
        # The circumradius in angular terms: if c is the circumcenter vector, and R is the Euclidean
        # distance from c to each vertex, then the angular distance theta satisfies:
        # R = 2 * sin(theta/2) * |c_perp| ... 
        # Actually, the angular circumradius is arccos(c . v0 / (|c| * 1)) if c is not zero.
        # But c may not be on the unit sphere.
        
        # The angular distance from the direction of c to v0 is arccos(c . v0 / (|c| * 1)).
        # This is the circumradius in angular terms.
        
        if np.linalg.norm(c) < 1e-12:
            continue
        
        c_norm = c / np.linalg.norm(c)
        cos_theta = np.dot(c_norm, v0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        
        # Check if the cap is empty: no other point is strictly inside the cap.
        # A point y is inside the cap if y . c_norm > c_norm . v0 (i.e., closer to c_norm than v0 is).
        # Actually, the cap boundary is the plane through the facet. Points on the sphere with
        # y . c_norm > c . v0 / |c| are inside the cap.
        
        # Wait, I need to be more careful. The circumradius of the facet is the angular distance
        # from the pole of the facet to the vertices. The pole is c_norm. The angular radius is theta.
        # The cap is empty if no other point has angular distance < theta from c_norm.
        
        # But actually, for the covering radius, we want the maximum circumradius over all facets
        # that are "active" (empty caps). If a cap is not empty, it's not a facet of the covering.
        
        # For simplicity, let's just compute the max circumradius over all facets.
        # This might overestimate if some caps are not empty, but it's a valid upper bound.
        
        if theta > covering_radius:
            covering_radius = theta
    
    return covering_radius


def _refine(points, max_iters=50, tol=1e-10):
    """Iteratively refine the configuration by moving each point to the center of its 
    Voronoi cell (the circumcenter of its Delaunay cell) to minimize the covering radius."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    for it in range(max_iters):
        max_delta = 0.0
        for i in range(n):
            # Find the Delaunay cell of point i: the set of points whose Voronoi cells share a face with i's cell
            # This is equivalent to finding all facets of the convex hull that contain point i.
            # But that's expensive. Instead, use a simpler heuristic:
            # Move point i in the direction of the "worst" direction (the direction of maximum distance to nearest point).
            
            # Compute the angular distance from each other point to point i
            # The covering radius contribution of point i is the maximum angular distance
            # from any point in its Voronoi cell to point i.
            
            # Approximation: find the direction on the sphere farthest from point i among the 
            # directions to other points' midpoints, or use the circumcenter of the Delaunay cell.
            
            # For now, use a simple gradient-free approach:
            # The force on point i is the sum of contributions from nearby points.
            # This is complex, so let's use a different strategy.
            
            pass
        
        # Alternative: use the fact that at optimum, each point is at the circumcenter
        # of its Delaunay cell. Let's implement this properly.
        
        # Compute convex hull to find Delaunay cells
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the facets that contain it
        point_facets = [[] for _ in range(n)]
        for f_idx, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_facets[idx].append(f_idx)
        
        for i in range(n):
            facets = point_facets[i]
            if len(facets) < 3:
                continue
            
            # The Delaunay cell of point i is the intersection of the half-spaces defined by the facets
            # containing i. The circumcenter of this cell is the point equidistant from all vertices
            # of the cell.
            
            # For a point on the sphere, the Delaunay cell is a spherical polygon. The circumcenter
            # is the point on the sphere equidistant from all vertices of the polygon.
            
            # This is complex. Let's use a simpler heuristic:
            # Move point i towards the centroid of its neighboring points' midpoints.
            
            # Find all points j that share a facet with i
            neighbors = set()
            for f_idx in facets:
                for idx in hull.simplices[f_idx]:
                    if idx != i:
                        neighbors.add(idx)
            
            if len(neighbors) == 0:
                continue
            
            # The new position should be the circumcenter of the spherical polygon formed by
            # the midpoints of the edges from i to its neighbors.
            # Approximate by moving i towards the average of the neighbor directions.
            
            neighbor_pts = points[list(neighbors)]
            # Weighted average: points closer to i should have more influence
            dists = np.array([np.linalg.norm(neighbor_pts[k] - points[i]) for k in range(len(neighbor_pts))])
            weights = 1.0 / (dists + 1e-10)
            weights /= np.sum(weights)
            
            new_pos = np.average(neighbor_pts, axis=0, weights=weights)
            new_pos = new_pos / np.linalg.norm(new_pos)
            
            delta = np.linalg.norm(new_pos - points[i])
            if delta > max_delta:
                max_delta = delta
            
            points[i] = new_pos
        
        if max_delta < tol:
            break
    
    return points


def construct():
    # Start with a Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using iterative circumcenter method
    points = _refine(points, max_iters=100, tol=1e-12)
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1)[:, None]
    
    return points
# EVOLVE-BLOCK-END
