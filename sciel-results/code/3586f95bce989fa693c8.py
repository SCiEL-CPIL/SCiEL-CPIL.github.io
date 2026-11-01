"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_TIME_LIMIT = 110.0  # safety margin below 120s wall-clock


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the
    sphere to its nearest configuration point. This equals the largest circumradius
    of the spherical Voronoi cells, which can be found via the convex hull.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0  # degenerate: return a large value

    # For each facet of the convex hull, compute the circumradius of the spherical cap.
    # The circumradius of a spherical polygon (facet) is the angular distance from
    # the center of the circumscribed circle to the vertices.
    # For a planar facet with vertices v_i on the unit sphere, the circumcenter
    # (in the plane) projected back to the sphere gives the farthest point.
    # The covering radius contribution from facet f is the angular distance from
    # the facet's spherical circumcenter to any of its vertices.

    max_radius_deg = 0.0
    facets = hull.simplices

    for simplex in facets:
        verts = pts[simplex]
        # Find the circumcenter of this triangle on the sphere.
        # The normal to the plane of the triangle:
        a, b, c = verts[0], verts[1], verts[2]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm

        # The circumcenter of the triangle in 3D lies along the normal direction.
        # For points on the unit sphere, the spherical circumcenter is where the
        # plane perpendicular to the sphere's radius at the circumcenter intersects
        # the sphere. We can compute the circumradius as the angular distance from
        # the "pole" of the facet to each vertex.
        #
        # Alternative: the angular circumradius R satisfies cos(R) = dot(circumcenter, vertex).
        # The circumcenter of the spherical triangle is the point on the sphere equidistant
        # from all three vertices. It lies along the direction of the normal to the plane
        # containing the three vertices, but we need to find the correct scaling.
        #
        # Actually, for a triangle on the unit sphere with vertices a, b, c,
        # the spherical circumcenter O satisfies |O-a|_angular = |O-b|_angular = |O-c|_angular.
        # O = n / (n . a) where n is the plane normal (if n . a > 0).
        # Let's verify: O . a = (n . a) / (n . a) = 1... that's not right.
        #
        # Let me think more carefully. The circumcenter of the planar triangle (in its plane)
        # projected onto the sphere gives the spherical circumcenter.
        #
        # Simpler approach: the circumradius of the spherical triangle equals the angular
        # distance from the pole (normal direction) to a vertex, adjusted.
        # Actually, the farthest point from all three vertices on the sphere within the
        # spherical triangle is the spherical circumcenter. The angular distance from
        # the circumcenter to any vertex is the circumradius.
        #
        # Let's compute it directly: find point O on sphere such that O.a = O.b = O.c = cos(R).
        # O is in the direction of the normal to the plane abc.
        # Let d = n . a (which equals n . b = n . c since a,b,c are in the plane).
        # Then O = n / d (normalized to unit length if d != 0).
        # O . a = (n . a) / d = d / d = 1. That gives R = 0, which is wrong.
        #
        # I'm confusing myself. Let me use a different approach.
        # The circumradius of the spherical triangle can be computed from the side lengths.
        # Or, more simply, the covering radius of the facet is the radius of the smallest
        # spherical cap centered at some point that covers all three vertices.
        # That's the spherical circumradius.
        #
        # For three points on the unit sphere, the spherical circumradius R satisfies:
        # cos(R) = (cos(a) + cos(b) + cos(c) - cos(a-b) - cos(b-c) - cos(c-a)) / ... 
        # This is getting complicated. Let me just use the fact that the circumcenter
        # of the spherical triangle lies along the direction of the cross product, and
        # compute R as the angular distance from that circumcenter to any vertex.

        # The circumcenter direction is the normalized normal, but we need the right sign
        # and the actual circumcenter is not simply n. Let me compute it properly.
        #
        # For a triangle with vertices on the unit sphere, the spherical circumcenter
        # is the intersection of the sphere with the line through the origin in the
        # direction of the normal to the plane of the triangle. But the normal alone
        # doesn't give the circumcenter; we need to account for the fact that the
        # circumcenter must be equidistant from all three vertices.
        #
        # Let O = t * n where t is a scalar. We need O to be on the unit sphere:
        # |t * n| = 1 => t = 1 (since |n| = 1). So O = n or O = -n.
        # O . a = n . a = d. So cos(R) = d, R = arccos(d).
        # But wait, this assumes the circumcenter is exactly at the pole, which is only
        # true for equilateral triangles. For general triangles, the circumcenter is
        # NOT at the pole.
        #
        # OK, I need to compute the spherical circumcenter properly.
        # The spherical circumcenter C satisfies:
        # C . a = C . b = C . c = cos(R)
        # |C| = 1
        #
        # This is a system of 4 equations (3 dot products + normalization) with 4 unknowns
        # (Cx, Cy, Cz, R). We can solve it.
        #
        # From C . a = C . b: C . (a - b) = 0, so C is perpendicular to (a-b).
        # Similarly C . (b - c) = 0.
        # So C is perpendicular to both (a-b) and (b-c), meaning C is parallel to
        # (a-b) x (b-c) = the normal n.
        # So C = ±n / |n| = ±n (since n is already normalized).
        # Then C . a = ±n . a = ±d.
        # So cos(R) = ±d, and we pick the sign that gives the smaller R (closer to 0).
        # Actually, we want the circumcenter to be on the same side as the triangle,
        # so we pick the sign such that C . a > 0 (assuming the triangle is in the
        # "positive" hemisphere relative to n).
        #
        # Wait, this derivation shows that the circumcenter IS at the pole! Let me
        # re-examine. C . (a-b) = 0 means C is in the null space of (a-b), i.e., C
        # is perpendicular to a-b. Similarly for b-c. The only vector perpendicular to
        # both a-b and b-c is the normal n. So yes, C must be parallel to n.
        #
        # But this can't be right for a non-equilateral triangle. Let me check with a
        # simple example: a = (1,0,0), b = (0,1,0), c = (0,0,1).
        # a-b = (1,-1,0), b-c = (0,1,-1).
        # (a-b) x (b-c) = (-1,-1,-1). Normalized: (-1,-1,-1)/sqrt(3).
        # C = (-1,-1,-1)/sqrt(3). C . a = -1/sqrt(3). R = arccos(-1/sqrt(3)) ≈ 125.3°.
        # But the circumradius of this right spherical triangle should be 90° (the
        # circumcenter is at (1,1,1)/sqrt(3), and the distance to (1,0,0) is arccos(1/sqrt(3)) ≈ 54.7°).
        #
        # I see my error. The circumcenter is NOT at the pole. The condition C . (a-b) = 0
        # is wrong. What we need is |C - a|_angular = |C - b|_angular, which means
        # C . a = C . b, i.e., C . (a - b) = 0. That part is correct.
        # But for a=(1,0,0), b=(0,1,0), c=(0,0,1):
        # C . (1,-1,0) = 0 => Cx = Cy
        # C . (0,1,-1) = 0 => Cy = Cz
        # So Cx = Cy = Cz. With |C|=1, C = (1,1,1)/sqrt(3) or (-1,-1,-1)/sqrt(3).
        # C . a = 1/sqrt(3). R = arccos(1/sqrt(3)) ≈ 54.7°. That's correct!
        #
        # So my earlier calculation of the cross product was wrong. Let me redo:
        # (a-b) x (b-c) = (1,-1,0) x (0,1,-1)
        # = (-1*-1 - 0*1, 0*0 - 1*-1, 1*1 - (-1)*0)
        # = (1, 1, 1). So the normal is (1,1,1)/sqrt(3), not (-1,-1,-1).
        # I made a sign error. So C = (1,1,1)/sqrt(3) is correct.
        #
        # Great, so the circumcenter is indeed at the pole (in the direction of the normal),
        # and R = arccos(|n . a|) where we take the absolute value to get the acute angle.
        # Actually, we should take n . a and if it's negative, flip n.

        d = np.dot(n, a)
        if d < 0:
            n = -n
            d = -d
        # d should be <= 1 (since a is on unit sphere and n is unit)
        d = np.clip(d, -1.0, 1.0)
        R = np.arccos(d)
        R_deg = np.degrees(R)
        if R_deg > max_radius_deg:
            max_radius_deg = R_deg

    return max_radius_deg


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lloyd_relax(pts, max_iters=200, tol=1e-6):
    """
    Lloyd's algorithm (Voronoi relaxation) on the sphere.
    For each point, compute the centroid of its spherical Voronoi cell and move
    the point to that centroid. This minimizes the sum of squared distances to
    Voronoi vertices, which helps reduce the covering radius.
    """
    pts = _normalize(pts)
    for it in range(max_iters):
        # Compute spherical Voronoi cells via convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # For each point, find the facets (triangles) that contain it as a vertex.
        # The spherical Voronoi cell of point i is the set of points on the sphere
        # closer to i than to any other point. The centroid of this cell is the
        # direction of the sum of the normals of the adjacent facets, weighted by
        # facet area, or more simply, we can approximate it as the normalized sum
        # of the midpoints of the edges of the Voronoi cell.
        #
        # A simpler approach: for each point, find all facets that include it,
        # compute the centroid of each facet (as a direction on the sphere),
        # and average them.
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            # Find all facets containing point i
            mask = np.any(hull.simplices == i, axis=1)
            facets = hull.simplices[mask]
            if len(facets) == 0:
                new_pts[i] = pts[i]
                continue
            
            # For each facet, compute its centroid direction
            centroids = []
            for simplex in facets:
                verts = pts[simplex]
                # The centroid of the spherical triangle is approximately
                # the normalized sum of the vertices
                centroid = np.sum(verts, axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 1e-10:
                    centroids.append(centroid / c_norm)
            
            if len(centroids) == 0:
                new_pts[i] = pts[i]
            else:
                avg = np.mean(centroids, axis=0)
                avg_norm = np.linalg.norm(avg)
                if avg_norm > 1e-10:
                    new_pts[i] = avg / avg_norm
                else:
                    new_pts[i] = pts[i]
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < tol:
            break

    return _normalize(pts)


def _random_perturbation(pts, sigma=0.05):
    """Add random Gaussian perturbation and renormalize."""
    noise = np.random.randn(N, 3) * sigma
    return _normalize(pts + noise)


def construct():
    """
    Construct 130 points on the unit sphere to minimize the covering radius.
    Uses multi-start with Lloyd relaxation and simulated annealing-like perturbation.
    """
    start_time = time.time()
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere()
    best_radius = _covering_radius(best_pts)
    
    rng = np.random.default_rng(42)
    
    # Multi-start with different initial configurations
    n_starts = 10
    for s in range(n_starts):
        if time.time() - start_time > _TIME_LIMIT - 10:
            break
        
        if s == 0:
            pts = _fibonacci_sphere()
        else:
            # Random initial configuration
            pts = rng.standard_normal((N, 3))
            pts = _normalize(pts)
        
        # Lloyd relaxation
        pts = _lloyd_relax(pts, max_iters=100)
        radius = _covering_radius(pts)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
        
        # Local search: perturb and re-relax
        for local_iter in range(30):
            if time.time() - start_time > _TIME_LIMIT - 5:
                break
            
            sigma = 0.1 / (1.0 + local_iter * 0.1)
            perturbed = _random_perturbation(pts, sigma=sigma)
            relaxed = _lloyd_relax(perturbed, max_iters=50)
            r = _covering_radius(relaxed)
            
            if r < radius:
                radius = r
                pts = relaxed
                if radius < best_radius:
                    best_radius = radius
                    best_pts = pts.copy()
    
    return _normalize(best_pts)


# EVOLVE-BLOCK-START
# (The above functions are the main content; this block is kept for compatibility.)
# EVOLVE-BLOCK-END
