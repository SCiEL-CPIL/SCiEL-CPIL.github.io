"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees for a set of points on the sphere."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid or degenerate
        
    # Get the facets (simplices) of the convex hull
    facets = hull.simplices
    
    # For each facet, compute the circumcenter and circumradius
    # The covering radius is the max over all facets of the angular distance
    # from the facet's circumcenter to the farthest vertex of that facet
    # Actually, for a spherical cap covering problem, the covering radius is the
    # maximum over all points y on the sphere of min_i angle(y, p_i).
    # This equals the maximum circumradius of the spherical triangles (facets).
    
    # For each triangular facet (i, j, k), the circumcenter on the sphere
    # is the point equidistant from i, j, k. The angular distance from this
    # circumcenter to any of i, j, k is the circumradius of that spherical triangle.
    
    max_radius = 0.0
    for fac in facets:
        i, j, k = fac
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # The circumcenter of a spherical triangle is the normalized sum
        # of the three vertices weighted by sin of opposite angles, but
        # a simpler approach: the circumcenter is the unit vector in the
        # direction of the cross product-based normal... 
        # Actually, for a triangle on the unit sphere, the circumcenter
        # (in the sense of the center of the circumscribed circle on the sphere)
        # can be found as follows: it's the point equidistant from all three.
        # The circumcenter direction is proportional to:
        #   (pj x pk) * |pi - proj| ... this gets complicated.
        # 
        # Simpler: the angular distance from the circumcenter to vertex i is
        # the same for all three vertices. We can compute it from the edge lengths.
        # For a spherical triangle with side lengths a, b, c (opposite to A, B, C),
        # the circumradius R satisfies: sin(R) = ... 
        # 
        # Alternative: compute the circumcenter as the unit vector c such that
        # c · pi = c · pj = c · pk = cos(R)
        # This means c is in the span of pi, pj, pk and satisfies:
        #   c · pi = c · pj = c · pk
        # Let c = alpha*pi + beta*pj + gamma*pk
        # Then: alpha + beta*(pj·pi) + gamma*(pk·pi) = alpha*(pi·pj) + beta + gamma*(pk·pj) = ...
        # This is a linear system. Let G be the Gram matrix [pi·pi, pi·pj, pi·pk; ...]
        # We want G * [alpha, beta, gamma]^T = [1, 1, 1]^T (up to scale)
        # Then c = alpha*pi + beta*pj + gamma*pk, normalize, and R = arccos(c·pi)
        
        G = np.array([
            [np.dot(pi, pi), np.dot(pi, pj), np.dot(pi, pk)],
            [np.dot(pj, pi), np.dot(pj, pj), np.dot(pj, pk)],
            [np.dot(pk, pi), np.dot(pk, pj), np.dot(pk, pk)]
        ])
        ones = np.ones(3)
        
        try:
            coeffs = np.linalg.solve(G, ones)
        except np.linalg.LinAlgError:
            continue
            
        c = coeffs[0]*pi + coeffs[1]*pj + coeffs[2]*pk
        cnorm = np.linalg.norm(c)
        if cnorm < 1e-10:
            continue
        c = c / cnorm
        
        # The angular distance from c to pi (and pj, pk)
        cosR = np.clip(np.dot(c, pi), -1.0, 1.0)
        R = np.arccos(cosR)
        if R > max_radius:
            max_radius = R
    
    return np.degrees(max_radius)


def _lloyd_iter(pts, max_iters=200, tol=1e-10):
    """Lloyd's algorithm for spherical Voronoi: move each point to the centroid of its Voronoi cell."""
    pts = pts.copy()
    for it in range(max_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Compute Voronoi cell centroids using the convex hull
        # For each point, find the adjacent facets and compute the spherical centroid
        # A simpler approach: for each point, find its Voronoi cell vertices
        # which are the circumcenters of adjacent facets.
        
        # Build adjacency: for each point, which facets contain it
        point_facets = [[] for _ in range(len(pts))]
        for idx, fac in enumerate(hull.simplices):
            for p in fac:
                point_facets[p].append(idx)
        
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            # Collect all vertices of the Voronoi cell for point i
            # These are the circumcenters of all facets containing i
            cell_vertices = []
            for fac_idx in point_facets[i]:
                fac = hull.simplices[fac_idx]
                pi = pts[fac[0]]
                pj = pts[fac[1]]
                pk = pts[fac[2]]
                
                G = np.array([
                    [np.dot(pi, pi), np.dot(pi, pj), np.dot(pi, pk)],
                    [np.dot(pj, pi), np.dot(pj, pj), np.dot(pj, pk)],
                    [np.dot(pk, pi), np.dot(pk, pj), np.dot(pk, pk)]
                ])
                ones = np.ones(3)
                try:
                    coeffs = np.linalg.solve(G, ones)
                except np.linalg.LinAlgError:
                    continue
                c = coeffs[0]*pi + coeffs[1]*pj + coeffs[2]*pk
                cnorm = np.linalg.norm(c)
                if cnorm > 1e-10:
                    cell_vertices.append(c / cnorm)
            
            if len(cell_vertices) > 0:
                centroid = np.mean(cell_vertices, axis=0)
                cnorm = np.linalg.norm(centroid)
                if cnorm > 1e-10:
                    new_pts[i] = centroid / cnorm
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < tol:
            break
    
    return pts


def _fibonacci_sphere(n):
    """Generate n points on the sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    best_pts = None
    best_radius = np.pi  # 180 degrees, worst case
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Run Lloyd's algorithm from the Fibonacci start
    while time.time() < deadline - 30.0:
        pts = _lloyd_iter(pts, max_iters=100)
        radius = _covering_radius_deg(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
        
        # Check if we've converged
        # Run another Lloyd iteration and see if it changes much
        pts2 = _lloyd_iter(pts, max_iters=10)
        if np.linalg.norm(pts2 - pts) < 1e-8:
            break
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        pts = _random_sphere(N, rng)
        pts = _lloyd_iter(pts, max_iters=50)
        radius = _covering_radius_deg(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final polish: run more Lloyd iterations on the best configuration
    if time.time() < deadline - 10.0:
        best_pts = _lloyd_iter(best_pts, max_iters=200, tol=1e-12)
    
    # Ensure all points are normalized
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
