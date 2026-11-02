"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (max angular distance to nearest point) in radians.
    
    Uses the convex hull: the covering radius is the maximum circumradius of the
    facets of the convex hull (in angular terms, the angular distance from the
    centroid of each facet's great-circle triangle to its vertices).
    
    Actually, for sphere covering: the covering radius is the maximum over all
    points y on the sphere of the angular distance to the nearest chosen point.
    This equals the maximum circumradius of the spherical Voronoi cells, which
    corresponds to the angular distance from each hull facet's circumcenter (on
    the sphere) to its three vertices.
    
    We compute: for each triangular facet of the convex hull, find the point on
    the sphere that is farthest from all three vertices (the circumcenter of the
    spherical triangle), and the angular distance to each vertex. The maximum
    over all facets is the covering radius.
    
    A simpler and robust approach: for each facet (triangle on convex hull),
    the farthest point from the three vertices on the sphere is the point
    opposite to the triangle's centroid direction, or more precisely, we can
    compute the circumradius of the spherical triangle.
    
    Actually, the cleanest method: the covering radius is the maximum, over all
    hull facets, of the angular distance from the facet's "pole" (normalized
    normal vector pointing inward) to the nearest vertex of that facet.
    
    Wait - let me think more carefully. The convex hull of N points on the unit
    sphere has facets. Each facet is a triangle (for generic position). The
    spherical Voronoi cell of each point is bounded by arcs. The covering radius
    is the maximum distance from any point on the sphere to its nearest data
    point. This occurs at the circumcenter of some spherical Voronoi cell.
    
    Equivalently: for each triangular face of the convex hull with vertices
    p1, p2, p3, the point on the sphere farthest from all three is in the
    direction of the normalized cross product or the normalized sum... 
    
    Actually, the circumcenter of a spherical triangle: the point equidistant
    from all three vertices. For a triangle with vertices a, b, c on the unit
    sphere, the circumcenter (in terms of the point on the sphere equidistant
    from a, b, c) is in the direction of (a x b) x (a x c) normalized, or
    equivalently the direction that makes equal angles with a, b, c.
    
    A simpler computational approach: for each hull facet, compute the normal
    vector n (normalized, pointing outward). The point on the sphere closest to
    the facet's "center" in the spherical sense... 
    
    Let me use a different, well-known approach: the covering radius is the
    maximum over all facets of the convex hull of the angular distance from
    the facet centroid (normalized) to the nearest vertex. But this isn't
    exactly right either.
    
    The correct approach: For each triangular facet with vertices a, b, c on
    the unit sphere, find the point x on the unit sphere that maximizes
    min(angle(x,a), angle(x,b), angle(x,c)). This is the circumcenter of the
    spherical triangle. The angular distance from this x to a (or b or c) is
    the circumradius of that spherical triangle.
    
    For a spherical triangle, the circumcenter can be found as follows:
    The point x = αa + βb + γc (normalized) such that x·a = x·b = x·c.
    This gives us a system of equations.
    
    Alternatively: the circumcenter direction is proportional to:
    a * (b×c) + b * (c×a) + c * (a×b) ... no, that's not right either.
    
    Let me use the standard formula. For three points a, b, c on the unit sphere,
    the circumcenter (point on sphere equidistant from all three) is:
    x ∝ (a × b) × (b × c) ... 
    
    Actually, the simplest robust method: solve for x on the unit sphere
    minimizing max(angle to a, angle to b, angle to c). By symmetry of the
    circumcenter, x·a = x·b = x·c = cos(R) where R is the circumradius.
    
    Let x = w1*a + w2*b + w3*c. Then:
    x·a = w1 + w2*(b·a) + w3*(c·a) = cos(R)
    x·b = w1*(a·b) + w2 + w3*(c·b) = cos(R)
    x·c = w1*(a·c) + w2*(b·c) + w3 = cos(R)
    
    And |x| = 1.
    
    This is getting complex. Let me use a simpler but valid approach:
    
    The covering radius equals the maximum, over all points y on the sphere,
    of the distance to the nearest data point. This is equal to the maximum
    circumradius of the spherical Delaunay triangles, which correspond to the
    facets of the convex hull.
    
    For each hull facet with vertices a, b, c, the circumradius R of the
    spherical triangle can be computed. If the triangle is "small" (area < π),
    the circumcenter is the unique point equidistant from a, b, c on the
    same side.
    
    Formula: For a spherical triangle with side lengths A, B, C (angular
    distances between vertices), the circumradius R satisfies:
    tan(R) = (tan(A/2)*tan(B/2)*tan(C/2)) / 
              (tan(A/2)*tan(B/2) + tan(B/2)*tan(C/2) + tan(C/2)*tan(A/2))
    
    Wait, that's not quite right. Let me look up the correct formula.
    
    For a spherical triangle with angles α, β, γ and side lengths a, b, c:
    The circumradius R is given by:
    tan(R) = (sin(a) * sin(b) * sin(c)) / (4 * Δ)
    where Δ is the area of the spherical triangle? No...
    
    Actually, a simpler approach that's numerically stable:
    For each facet, compute the normal vector n = (b-a) × (c-a), normalized.
    The circumcenter of the spherical triangle lies along the direction:
    x = normalize(n * something + (a+b+c) * something)
    
    Let me just use the direct approach: for each hull facet, the farthest
    point from all three vertices on the sphere is found by:
    x = normalize(a + b + c) if the triangle is acute, otherwise it's at a
    vertex. But for covering radius purposes, we want the maximum.
    
    Actually, the most robust and simple approach: 
    The covering radius is the maximum over all hull facets of the angular
    distance from the normalized facet normal (pointing toward the sphere
    center, i.e., inward) to the nearest vertex of that facet.
    
    Wait no. Let me think geometrically. The convex hull of points on the unit
    sphere. Each facet is a flat triangle in 3D. The "spherical" Voronoi
    diagram is different from the 3D Voronoi diagram.
    
    Key insight: A point y on the sphere is equidistant (in angular distance)
    to three data points a, b, c if and only if y is in the direction of
    (a+b+c) normalized (for the "incenter" case) or more generally.
    
    Actually, y·a = y·b = y·c means y is in the null space of the matrix
    [[a-b], [b-c]] (i.e., y·(a-b)=0 and y·(b-c)=0), so y is parallel to
    (a-b) × (b-c) = a×b + b×c + c×a... let me verify:
    (a-b) × (b-c) = a×b - a×c - b×b + b×c = a×b - a×c + b×c
    = a×b + b×c - a×c = a×b + b×c + c×a
    
    So the circumcenter direction is:
    x = normalize((a-b) × (b-c)) = normalize(a×b + b×c + c×a)
    
    And the circumradius is the angular distance from x to a (or b or c).
    
    But we need to make sure x points in the right direction (the one that
    gives the maximum minimum distance, not the minimum). There are two
    antipodal points; we want the one that's farthest from all three, which
    is the one where x·a = x·b = x·c is minimized (most negative) if we're
    looking at the "far" side, or... 
    
    Actually for the covering radius, we want the point on the sphere
    farthest from its nearest neighbor. This occurs at circumcenters of
    spherical triangles where the circumcenter is "outside" the triangle in
    some sense. 
    
    For a convex hull facet, the relevant circumcenter is the one on the
    same side as the facet's outward normal (or inward?). 
    
    Let me just compute both antipodal points and take the one with larger
    minimum angular distance to the three vertices.
    
    Let me implement this properly.
    """
    points = np.asarray(points, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute convex hull
    hull = ConvexHull(points)
    facets = hull.simplices  # shape (F, 3)
    
    max_circumradius = 0.0
    
    for tri in facets:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        
        # Circumcenter direction
        cross1 = np.cross(a - b, b - c)
        norm_cross1 = np.linalg.norm(cross1)
        
        if norm_cross1 < 1e-12:
            continue
        
        x = cross1 / norm_cross1
        
        # Compute angular distances from x to a, b, c
        dot_a = np.clip(np.dot(x, a), -1.0, 1.0)
        dot_b = np.clip(np.dot(x, b), -1.0, 1.0)
        dot_c = np.clip(np.dot(x, c), -1.0, 1.0)
        
        ang_a = np.arccos(dot_a)
        ang_b = np.arccos(dot_b)
        ang_c = np.arccos(dot_c)
        
        min_ang = min(ang_a, ang_b, ang_c)
        
        # Also check the antipodal point
        x_neg = -x
        dot_a2 = np.clip(np.dot(x_neg, a), -1.0, 1.0)
        dot_b2 = np.clip(np.dot(x_neg, b), -1.0, 1.0)
        dot_c2 = np.clip(np.dot(x_neg, c), -1.0, 1.0)
        
        ang_a2 = np.arccos(dot_a2)
        ang_b2 = np.arccos(dot_b2)
        ang_c2 = np.arccos(dot_c2)
        
        min_ang2 = min(ang_a2, ang_b2, ang_c2)
        
        # The covering radius contribution from this facet is the max of the two
        circumradius = max(min_ang, min_ang2)
        
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    
    return max_circumradius


def _pairwise_angular_distances(points):
    """Compute pairwise angular distances."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    dots = np.clip(points @ points.T, -1.0, 1.0)
    return np.arccos(dots)


def _objective(points):
    """The covering radius in radians (lower is better)."""
    return _covering_radius(points)


def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_score = _covering_radius(best_pts)
    
    # Local optimization using gradient-free approach
    # We'll use a simple iterative method: for each point, move it in the
    # direction that reduces the maximum covering radius.
    
    # Use a steepest descent approach with random restarts
    np.random.seed(42)
    
    lr = 0.01
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Compute which facet gives the maximum covering radius
        # and which points are involved
        hull = ConvexHull(pts)
        facets = hull.simplices
        
        max_cr = 0.0
        worst_facet = None
        worst_point_idx = None
        worst_direction = None
        
        for tri in facets:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            cross1 = np.cross(a - b, b - c)
            norm_cross1 = np.linalg.norm(cross1)
            if norm_cross1 < 1e-12:
                continue
            x = cross1 / norm_cross1
            
            dot_a = np.clip(np.dot(x, a), -1.0, 1.0)
            dot_b = np.clip(np.dot(x, b), -1.0, 1.0)
            dot_c = np.clip(np.dot(x, c), -1.0, 1.0)
            
            ang_a = np.arccos(dot_a)
            ang_b = np.arccos(dot_b)
            ang_c = np.arccos(dot_c)
            
            min_ang = min(ang_a, ang_b, ang_c)
            
            x_neg = -x
            dot_a2 = np.clip(np.dot(x_neg, a), -1.0, 1.0)
            dot_b2 = np.clip(np.dot(x_neg, b), -1.0, 1.0)
            dot_c2 = np.clip(np.dot(x_neg, c), -1.0, 1.0)
            
            ang_a2 = np.arccos(dot_a2)
            ang_b2 = np.arccos(dot_b2)
            ang_c2 = np.arccos(dot_c2)
            
            min_ang2 = min(ang_a2, ang_b2, ang_c2)
            
            circumradius = max(min_ang, min_ang2)
            
            if circumradius > max_cr:
                max_cr = circumradius
                worst_facet = tri
                # Determine which of the two antipodal points is the worst
                if min_ang >= min_ang2:
                    worst_direction = x
                    # The point that is closest to x among a,b,c is the one to move
                    dists = [ang_a, ang_b, ang_c]
                else:
                    worst_direction = x_neg
                    dists = [ang_a2, ang_b2, ang_c2]
                
                # Move the point that is closest to the circumcenter
                closest_local = np.argmin(dists)
                worst_point_idx = tri[closest_local]
        
        if worst_point_idx is None:
            break
        
        # Move the worst point in the direction away from the circumcenter
        # (i.e., toward the circumcenter to reduce the gap)
        # Actually, we want to move the point toward the circumcenter to reduce
        # the covering radius
        p = pts[worst_point_idx].copy()
        # Direction on sphere: project worst_direction onto tangent plane at p
        proj = worst_direction - np.dot(worst_direction, p) * p
        proj_norm = np.linalg.norm(proj)
        if proj_norm > 1e-12:
            direction = proj / proj_norm
        else:
            # If already aligned, pick a random tangent direction
            rand_vec = np.random.randn(3)
            proj = rand_vec - np.dot(rand_vec, p) * p
            proj_norm = np.linalg.norm(proj)
            if proj_norm > 1e-12:
                direction = proj / proj_norm
            else:
                break
        
        # Small step
        new_p = p + lr * direction
        new_p = new_p / np.linalg.norm(new_p)
        pts[worst_point_idx] = new_p
        
        # Check if score improved
        new_score = _covering_radius(pts)
        if new_score < best_score:
            best_score = new_score
            best_pts = pts.copy()
        else:
            # Revert
            pts[worst_point_idx] = p
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-6:
                break
    
    # Final: return best points found
    return best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
# EVOLVE-BLOCK-END
