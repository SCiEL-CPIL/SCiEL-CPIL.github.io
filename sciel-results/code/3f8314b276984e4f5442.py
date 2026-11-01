"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _covering_radius(points):
    """Compute covering radius in radians: max over sphere of angular distance to nearest point.
    This equals the max circumradius of facets of the convex hull."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            facet_pts = points[simplex]
            # Compute circumcenter of the facet (triangle for 3D hull)
            # Use the formula: circumcenter of triangle with vertices a,b,c
            a, b, c = facet_pts
            # Circumcenter in 3D for a triangle
            # Using vector cross product method
            ab = b - a
            ac = c - a
            bc = c - b
            # Circumcenter O satisfies |O-a|^2 = |O-b|^2 = |O-c|^2
            # 2*O·(b-a) = |b|^2 - |a|^2
            # 2*O·(c-a) = |c|^2 - |a|^2
            # Also O = a + alpha*ab + beta*ac
            # Solve 2x2 system
            # (ab·ab)*alpha + (ab·ac)*beta = (|b|^2 - |a|^2)/2
            # (ac·ab)*alpha + (ac·ac)*beta = (|c|^2 - |a|^2)/2
            rhs1 = (np.dot(b, b) - np.dot(a, a)) / 2.0
            rhs2 = (np.dot(c, c) - np.dot(a, a)) / 2.0
            A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                          [np.dot(ac, ab), np.dot(ac, ac)]])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-12:
                # Degenerate, use centroid
                center = (a + b + c) / 3.0
            else:
                alpha = (rhs1 * A[1, 1] - rhs2 * A[0, 1]) / det
                beta = (rhs2 * A[0, 0] - rhs1 * A[1, 0]) / det
                center = a + alpha * ab + beta * ac
            # The circumradius is the distance from center to any vertex, but we need
            # the angular radius: the cap centered at center/|center| with angular radius
            # such that the facet lies on the boundary. Actually, the covering radius contribution
            # from this facet is the angle between the facet normal direction and the farthest point.
            # More precisely: the facet defines a plane. The point on the sphere farthest from all 
            # hull points within this facet's Voronoi region is along the direction of the circumcenter.
            # The angular distance from that direction to the nearest hull point is the circumradius angle.
            # For a facet with circumcenter C (not necessarily unit), the angular covering radius
            # contributed is arccos(max dot product of C/|C| with the facet vertices)? No.
            # Actually: the covering radius is the max over all points y on sphere of min_i angle(y, p_i).
            # The worst point for a facet is the point on the sphere in the direction of the facet's
            # circumcenter (projected to sphere). The angular distance from that point to the nearest
            # hull vertex is the circumradius angle.
            # For a triangle on the unit sphere, the circumradius angle is:
            # If circumcenter direction is n = C/|C|, then the angular distance to vertex a is arccos(n·a).
            # But actually, the circumradius of the spherical triangle is what we want.
            # Let's compute it directly: the angular radius r such that the cap of radius r centered at n
            # contains the triangle. That's max over vertices of angle(n, v).
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-12:
                # Degenerate
                continue
            n = center / norm_c
            # Angular distance from n to each vertex
            dots = np.array([np.dot(n, a), np.dot(n, b), np.dot(n, c)])
            # Clip to [-1, 1]
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            facet_r = np.max(angles)
            max_r = max(max_r, facet_r)
        return max_r
    except Exception:
        # Fallback: sample-based estimate
        # Generate random points and compute max min-angle
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(10000, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        # Compute min angle for each sample
        dots = samples @ points.T  # (10000, 50)
        dots = np.clip(dots, -1.0, 1.0)
        min_angles = np.arccos(np.max(dots, axis=1))
        return np.max(min_angles)


def _lloyd_relaxation(points, max_iter=500, tol=1e-10):
    """Lloyd's relaxation: move each point to centroid of its Voronoi cell on the sphere."""
    from scipy.spatial import ConvexHull
    points = points.copy()
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        # For each point, find its Voronoi cell: the set of facets adjacent to it
        # The centroid of the Voronoi cell on the sphere is the normalized sum of
        # the circumcenters of adjacent facets weighted by... actually, for Lloyd's method
        # on the sphere, we move to the centroid of the Voronoi cell.
        # A simpler approach: for each point, collect all hull vertices connected to it
        # via edges, and also the circumcenters of adjacent facets.
        # Standard Lloyd: new position = centroid of Voronoi cell.
        # For a point on the sphere, the Voronoi cell is a spherical polygon.
        # We approximate by collecting the circumcenters of all facets sharing that point,
        # and averaging them (then normalizing).
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            # Find all facets that include point i
            adjacent_facets = []
            for simplex in hull.simplices:
                if i in simplex:
                    adjacent_facets.append(simplex)
            if not adjacent_facets:
                new_points[i] = points[i]
                continue
            # For each adjacent facet, compute circumcenter
            centers = []
            for simplex in adjacent_facets:
                facet_pts = points[simplex]
                a, b, c = facet_pts
                ab = b - a
                ac = c - a
                rhs1 = (np.dot(b, b) - np.dot(a, a)) / 2.0
                rhs2 = (np.dot(c, c) - np.dot(a, a)) / 2.0
                A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                              [np.dot(ac, ab), np.dot(ac, ac)]])
                det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                if abs(det) < 1e-12:
                    center = (a + b + c) / 3.0
                else:
                    alpha = (rhs1 * A[1, 1] - rhs2 * A[0, 1]) / det
                    beta = (rhs2 * A[0, 0] - rhs1 * A[1, 0]) / det
                    center = a + alpha * ab + beta * ac
                centers.append(center)
            # Average the circumcenters and normalize
            avg_center = np.mean(centers, axis=0)
            norm = np.linalg.norm(avg_center)
            if norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = avg_center / norm
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    return points


def _generate_initial_configurations(num_configs=5):
    """Generate multiple initial configurations."""
    configs = []
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Random configurations
    rng = np.random.default_rng(123)
    for _ in range(num_configs - 1):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    configs = _generate_initial_configurations(5)
    
    for config in configs:
        if time.time() > deadline - 20:
            break
        
        # Lloyd's relaxation
        relaxed = _lloyd_relaxation(config, max_iter=300, tol=1e-10)
        
        # Evaluate
        radius = _covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
        
        # If we have good time, try a second round of relaxation
        if time.time() < deadline - 30:
            relaxed2 = _lloyd_relaxation(relaxed, max_iter=200, tol=1e-12)
            radius2 = _covering_radius(relaxed2)
            if radius2 < best_radius:
                best_radius = radius2
                best_points = relaxed2.copy()
    
    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are normalized and non-zero
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
