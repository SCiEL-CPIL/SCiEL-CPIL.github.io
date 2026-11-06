"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _slerp(a, b, t):
    """Spherical linear interpolation between unit vectors a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-12:
        return _normalize(a + t * (b - a))
    s = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / s
    w2 = np.sin(t * theta) / s
    return _normalize(w1 * a + w2 * b)


def _facet_circumradius(p1, p2, p3):
    """
    Compute the angular circumradius of a spherical triangle with vertices p1, p2, p3 (unit vectors).
    The circumradius is the angular distance from the spherical circumcenter to each vertex.
    """
    p1 = _normalize(p1)
    p2 = _normalize(p2)
    p3 = _normalize(p3)

    # Edge lengths
    d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
    d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
    d31 = np.arccos(np.clip(np.dot(p3, p1), -1, 1))

    # Spherical circumradius formula:
    # R = arcsin( (sin(a/2) * sin(b/2) * sin(c/2)) / (sin(a/2)*sin(b/2)*sin(c/2) ... ) )
    # Actually, use the formula:
    # Let A, B, C be the angles of the spherical triangle.
    # Circumradius R satisfies: sin(R) = sin(a) sin(b) sin(c) / (4 * Area)
    # where Area is the spherical excess.
    # Simpler: use the formula R = arcsin( (sin(a/2) sin(b/2) sin(c/2)) / (sin(a/2) sin(b/2) sin(c/2) + ... ) )
    # Let's use a robust numerical approach: find the circumcenter by solving for a point equidistant from all three.

    # Alternative: The circumcenter c satisfies |c - p1|_s = |c - p2|_s = |c - p3|_s
    # which means dot(c, p1) = dot(c, p2) = dot(c, p3) = cos(R)
    # So c is in the null space of [p1-p2, p1-p3] in some sense.
    # c = (p1 + p2 + p3) / |p1+p2+p3| is NOT the circumcenter in general.
    # The circumcenter lies on the line through the origin and the vector that is a linear combination.
    # Specifically, c is proportional to the vector:
    # c = (p1 x p2) x (p2 x p3) ... no.
    # The circumcenter is along the direction:
    # n1 = p1 x p2, n2 = p2 x p3
    # The circumcenter direction is (n1 x n2) normalized? No.
    # Actually, the circumcenter c satisfies:
    # c . (p1 - p2) = 0  (in the tangent plane sense, but on sphere it's c.p1 = c.p2)
    # So c is orthogonal to (p1-p2) and (p2-p3) in the sense that c.p1 = c.p2 and c.p2 = c.p3.
    # This means c is in the direction of the cross product of (p1-p2) and (p2-p3)? No.
    # c . p1 = c . p2 => c . (p1 - p2) = 0
    # c . p2 = c . p3 => c . (p2 - p3) = 0
    # So c is orthogonal to both (p1-p2) and (p2-p3).
    # Thus c is parallel to (p1-p2) x (p2-p3).

    v1 = p1 - p2
    v2 = p2 - p3
    c_dir = np.cross(v1, v2)
    c_dir = _normalize(c_dir)

    # But we need to check: is this the correct direction? 
    # Let's verify: c_dir . p1 should equal c_dir . p2.
    # c_dir . (p1 - p2) = 0 by construction. Good.
    # But there are two antipodal points. We want the one that is the circumcenter of the smaller triangle.
    # The circumcenter should be on the same side as the triangle's interior.
    # We can check the angle: if the circumcenter is on the "wrong" side, the circumradius will be > pi/2.
    # Let's compute R for both c_dir and -c_dir and pick the one with smaller R.

    dot1 = np.dot(c_dir, p1)
    R1 = np.arccos(np.clip(dot1, -1, 1))

    dot2 = np.dot(-c_dir, p1)
    R2 = np.arccos(np.clip(dot2, -1, 1))

    # The circumradius is the smaller of the two (for the minor arc triangle)
    # But actually, for a triangle on the sphere, the circumcenter is unique if the triangle is not degenerate.
    # The correct circumcenter is the one where the angular distance to all three vertices is the same.
    # Both c_dir and -c_dir satisfy this. The "circumradius" of the facet in the context of covering
    # is the angular distance from the circumcenter to the vertices, but we want the one that is
    # the "center" of the cap. The cap is the set of points closer to the circumcenter than to any other vertex.
    # For the covering radius, we need the largest circumradius of the facets.
    # The circumradius should be the angular distance from the circumcenter to the vertices.
    # If R1 < pi/2, use R1; otherwise use R2 (which would be pi - R1).
    # Actually, the circumradius is min(R1, R2) because the cap radius is the angular radius of the cap,
    # which is the distance from the center to the boundary. The center is the circumcenter.
    # But wait: the circumcenter of a spherical triangle is the center of the circle passing through the three vertices.
    # The radius of that circle is the circumradius.
    # There are two circles through three points on a sphere (the two great circles' intersections... no).
    # Actually, there is a unique circle (small circle) through three non-collinear points on a sphere, unless they are on a great circle.
    # The center of that circle is the circumcenter, and there are two antipodal centers.
    # The circumradius is the same for both (angular distance from center to vertex).
    # So R1 and R2 should be complementary: R1 + R2 = pi? No.
    # If c_dir . p1 = cos(R1), then (-c_dir) . p1 = -cos(R1) = cos(pi - R1).
    # So R2 = pi - R1.
    # The circumradius of the circle is R1 if R1 <= pi/2, else pi - R1.
    # But for the covering problem, the "cap" around a facet has its center at the circumcenter,
    # and the radius is the angular distance from the circumcenter to the vertices.
    # We want the smallest such radius, so we take min(R1, pi - R1).

    R = min(R1, np.pi - R1)
    return R


def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the sphere.
    The covering radius is the maximum, over all points on the sphere, of the angular distance
    to the nearest point in the set. This equals the maximum circumradius of the facets of the convex hull.
    """
    points = np.array(points, dtype=np.float64)
    points = np.array([_normalize(p) for p in points])

    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to a grid-based estimate
        # Sample many points on the sphere and compute max distance to nearest
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 200)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        grid = np.array([_normalize(p) for p in grid])
        # For each grid point, find min angular distance to any configuration point
        # This is O(N*M) which is 50*20000 = 1e6, acceptable
        max_dist = 0.0
        for g in grid:
            d = np.arccos(np.clip(np.dot(points, g), -1, 1))
            max_dist = max(max_dist, np.min(d))
        return np.degrees(max_dist)

    # Get facet indices
    facets = hull.simplices
    max_circumradius = 0.0
    for f in facets:
        p1, p2, p3 = points[f[0]], points[f[1]], points[f[2]]
        R = _facet_circumradius(p1, p2, p3)
        max_circumradius = max(max_circumradius, R)

    return np.degrees(max_circumradius)


def _lloyd_step(points):
    """
    One step of Lloyd's algorithm for spherical covering:
    Move each point to the circumcenter of its Voronoi cell (approximated by the centroid of its Voronoi region).
    For simplicity, we use the centroid of the adjacent facets' circumcenters.
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)

    try:
        hull = ConvexHull(points)
    except Exception:
        return points

    # For each point, find the facets that contain it
    point_facets = [[] for _ in range(n)]
    for idx, f in enumerate(hull.simplices):
        for v in f:
            point_facets[v].append(idx)

    new_points = np.zeros_like(points)
    for i in range(n):
        if not point_facets[i]:
            new_points[i] = points[i]
            continue
        # Compute the circumcenter of each adjacent facet
        centers = []
        for fidx in point_facets[i]:
            f = hull.simplices[fidx]
            p1, p2, p3 = points[f[0]], points[f[1]], points[f[2]]
            # Circumcenter direction
            v1 = p1 - p2
            v2 = p2 - p3
            c_dir = np.cross(v1, v2)
            c_dir = _normalize(c_dir)
            # Choose the correct antipodal point
            dot1 = np.dot(c_dir, p1)
            if np.arccos(np.clip(dot1, -1, 1)) > np.pi / 2:
                c_dir = -c_dir
            centers.append(c_dir)

        if centers:
            centers = np.array(centers)
            # Average the centers and normalize
            avg = np.sum(centers, axis=0)
            new_points[i] = _normalize(avg)
        else:
            new_points[i] = points[i]

    return new_points


def _refine_worst_facet(points, max_iters=50):
    """
    Refine the configuration by focusing on the worst facet (largest circumradius).
    Move the three vertices of the worst facet to reduce its circumradius.
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)

    for _ in range(max_iters):
        try:
            hull = ConvexHull(points)
        except Exception:
            break

        # Find the worst facet
        worst_idx = -1
        worst_R = 0.0
        for idx, f in enumerate(hull.simplices):
            p1, p2, p3 = points[f[0]], points[f[1]], points[f[2]]
            R = _facet_circumradius(p1, p2, p3)
            if R > worst_R:
                worst_R = R
                worst_idx = idx

        if worst_idx < 0:
            break

        f = hull.simplices[worst_idx]
        i, j, k = f[0], f[1], f[2]

        # Try to move the vertices to reduce the circumradius
        # Use a simple gradient-like step: move each vertex toward the circumcenter
        p1, p2, p3 = points[i], points[j], points[k]
        v1 = p1 - p2
        v2 = p2 - p3
        c_dir = np.cross(v1, v2)
        c_dir = _normalize(c_dir)
        dot1 = np.dot(c_dir, p1)
        if np.arccos(np.clip(dot1, -1, 1)) > np.pi / 2:
            c_dir = -c_dir

        # Move each vertex a small step toward the circumcenter
        step = 0.01
        new_p1 = _normalize(p1 + step * (c_dir - p1))
        new_p2 = _normalize(p2 + step * (c_dir - p2))
        new_p3 = _normalize(p3 + step * (c_dir - p3))

        # Check if this improves the covering radius
        test_points = points.copy()
        test_points[i] = new_p1
        test_points[j] = new_p2
        test_points[k] = new_p3

        # Quick check: compute the circumradius of the modified facet
        R_new = _facet_circumradius(new_p1, new_p2, new_p3)
        if R_new < worst_R:
            points[i] = new_p1
            points[j] = new_p2
            points[k] = new_p3

    return points


def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return np.array([_normalize(p) for p in pts])


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # Initialize with Fibonacci spiral
    points = _fibonacci_spiral(N)

    # Phase 1: Lloyd's algorithm iterations
    for it in range(100):
        if time.time() > deadline:
            break
        new_points = _lloyd_step(points)
        # Check if we made progress
        if np.allclose(new_points, points, atol=1e-8):
            break
        points = new_points

    # Phase 2: Refine worst facets
    for it in range(20):
        if time.time() > deadline:
            break
        points = _refine_worst_facet(points, max_iters=10)

    # Phase 3: Final Lloyd iterations
    for it in range(50):
        if time.time() > deadline:
            break
        new_points = _lloyd_step(points)
        if np.allclose(new_points, points, atol=1e-8):
            break
        points = new_points

    # Ensure all points are normalized
    points = np.array([_normalize(p) for p in points])

    return points
# EVOLVE-BLOCK-END
