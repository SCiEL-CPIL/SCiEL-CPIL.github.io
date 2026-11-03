"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
DEADLINE = 120.0  # seconds, leave margin for evaluator overhead

def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME)

def _covering_radius(pts):
    """Compute the covering radius in degrees.
    Uses the convex hull: the covering radius is the max circumradius of facets
    whose circumscribed sphere contains no other points (empty cap condition).
    """
    pts = np.asarray(pts)
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-12):
        return np.inf
    pts = pts / norms[:, None]
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    # For each facet, compute the circumradius (angular distance from sphere center to facet plane)
    # The angular covering radius is the max over facets of arccos(n . c) where c is the unit normal
    # pointing outward and the facet center... Actually, for a point on the unit sphere, the distance
    # to the nearest of the 3 (or more) vertices of a facet is maximized at the circumcenter of the
    # spherical triangle. The circumradius of a spherical triangle is the angle from the center of
    # the circumscribed circle to a vertex.
    
    # Alternative: for each facet, compute the plane, find the point on the sphere farthest from
    # all three vertices within that facet. This is the circumcenter of the spherical triangle.
    
    max_r = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]  # shape (k, 3), k=3 typically
        # Compute the normal to the facet
        if len(v) >= 3:
            v0, v1, v2 = v[0], v[1], v[2]
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # Ensure normal points outward (away from origin)
            # The centroid of the facet
            centroid = np.mean(v, axis=0)
            if np.dot(n, centroid) < 0:
                n = -n
            
            # The circumcenter of the spherical triangle is along the normal direction
            # The angular distance from the circumcenter to each vertex is the circumradius
            # circumcenter = n (unit vector)
            # angular distance = arccos(n . v_i)
            cos_dists = np.abs(np.dot(v, n))
            # The circumradius is arccos(min cos_dist) since we want max distance
            # Actually the circumcenter on the sphere is at n, and the distance to vertex i is arccos(n . v_i)
            # But we need to verify it's the correct circumcenter (the one that gives equal distances)
            # For a spherical triangle, the circumcenter is the point equidistant from all 3 vertices.
            # If n is the normal, then n . v0 = n . v1 = n . v2 only if the triangle is isosceles...
            # Actually no. The circumcenter of a spherical triangle is NOT simply the normal to the plane.
            
            # Let me think again. The covering radius is the maximum over all points y on the sphere
            # of min_i angular_dist(y, p_i). The maximum occurs at a circumcenter of some facet of the
            # convex hull (of the points). The circumcenter of a spherical triangle with vertices a, b, c
            # is the point y such that angular_dist(y,a) = angular_dist(y,b) = angular_dist(y,c) and y
            # is in the spherical triangle.
            
            # For a plane through 3 points on the unit sphere, the circumcenter of the spherical triangle
            # is at the intersection of the sphere with the line perpendicular to the plane through the
            # circumcenter of the planar triangle.
            
            # Let's compute the planar circumcenter of the triangle v0, v1, v2, then normalize to get
            # the spherical circumcenter.
            
            # Planar circumcenter:
            ax, ay, az = v0
            bx, by, bz = v1
            cx, cy, cz = v2
            
            d = 2 * (ax * (by - cz) + bx * (cz - ay) + cx * (ay - by))
            if abs(d) < 1e-12:
                continue
            
            ux = ((ax**2 + ay**2 + az**2) * (by - cz) + (bx**2 + by**2 + bz**2) * (cz - ay) + (cx**2 + cy**2 + cz**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2 + az**2) * (cz - bx) + (bx**2 + by**2 + bz**2) * (ax - cz) + (cx**2 + cy**2 + cz**2) * (bx - ax)) / d
            uz = ((ax**2 + ay**2 + az**2) * (bx - cy) + (bx**2 + by**2 + bz**2) * (cy - ax) + (cx**2 + cy**2 + cz**2) * (ax - by)) / d
            
            u = np.array([ux, uy, uz])
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-12:
                continue
            u = u / u_norm
            
            # The circumcenter on the sphere is u (or -u, whichever is in the correct hemisphere)
            # Check which one is closer to the centroid
            if np.dot(u, centroid) < 0:
                u = -u
            
            # Angular distance from u to each vertex
            cos_d = np.clip(np.dot(v, u), -1.0, 1.0)
            ang_d = np.arccos(cos_d)
            r = np.max(ang_d)
            
            # Verify empty cap: no other point should be closer to u than to the vertices
            # i.e., all other points should be at angular distance >= r from u
            # Actually, the condition is that the cap of radius r centered at u contains no other points
            # other than the vertices. But since we're computing the max circumradius, we just need
            # that u is a valid circumcenter (equidistant from the 3 vertices).
            # The "empty cap" condition ensures this is actually a Voronoi vertex.
            
            # For now, just take the max circumradius of all facets
            if r > max_r:
                max_r = r
    
    return np.degrees(max_r)

def _lloyd_relax(pts, max_iters=500, tol=1e-10):
    """Lloyd relaxation: move each point to the centroid of its Voronoi cell."""
    pts = np.array(pts, dtype=float)
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        # Compute Voronoi cells using convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (set of facets adjacent to it)
        # The centroid of the Voronoi cell on the sphere is the normalized sum of the circumcenters
        # of the adjacent facets, or more simply, we can use the average of the facet normals.
        
        # Alternative approach: for each point, compute the spherical centroid of its Voronoi region.
        # A simpler approximation: move each point toward the centroid of its neighboring points.
        
        # Actually, let's use a simpler relaxation: for each point, compute the centroid of all
        # points in its Voronoi cell (approximated by the adjacent hull facets' circumcenters).
        
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            # Find all simplices (facets) that include point i
            adjacent = [s for s in hull.simplices if i in s]
            if len(adjacent) == 0:
                new_pts[i] = pts[i]
                continue
            
            # Compute circumcenter of each adjacent facet and average them
            circumcenters = []
            for s in adjacent:
                v = pts[s]
                v0, v1, v2 = v[0], v[1], v[2]
                ax, ay, az = v0
                bx, by, bz = v1
                cx, cy, cz = v2
                d = 2 * (ax * (by - cz) + bx * (cz - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax**2 + ay**2 + az**2) * (by - cz) + (bx**2 + by**2 + bz**2) * (cz - ay) + (cx**2 + cy**2 + cz**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2 + az**2) * (cz - bx) + (bx**2 + by**2 + bz**2) * (ax - cz) + (cx**2 + cy**2 + cz**2) * (bx - ax)) / d
                uz = ((ax**2 + ay**2 + az**2) * (bx - cy) + (bx**2 + by**2 + bz**2) * (cy - ax) + (cx**2 + cy**2 + cz**2) * (ax - by)) / d
                u = np.array([ux, uy, uz])
                u_norm = np.linalg.norm(u)
                if u_norm < 1e-12:
                    continue
                u = u / u_norm
                centroid = np.mean(v, axis=0)
                if np.dot(u, centroid) < 0:
                    u = -u
                circumcenters.append(u)
            
            if len(circumcenters) == 0:
                new_pts[i] = pts[i]
            else:
                centroid = np.mean(circumcenters, axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-12:
                    new_pts[i] = pts[i]
                else:
                    new_pts[i] = centroid / c_norm
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < tol:
            break
    
    return pts

def _optimize_from_start(pts0, max_time=30.0):
    """Run Lloyd relaxation and local optimization from a starting configuration."""
    t0 = time.time()
    pts = _lloyd_relax(pts0, max_iters=200)
    
    if time.time() - t0 > max_time * 0.7:
        return pts
    
    # Local optimization using gradient-free method on the sphere
    # Flatten the points and use Nelder-Mead or similar
    # This is expensive for 150 parameters, so let's do a few iterations of
    # moving each point slightly to reduce the covering radius
    
    # Simple local search: for each point, try small perturbations
    best_r = _covering_radius(pts)
    for it in range(50):
        if time.time() - t0 > max_time:
            break
        improved = False
        for i in range(N):
            if time.time() - t0 > max_time:
                break
            # Try small perturbations in random directions
            for trial in range(20):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                for step in [0.01, 0.005, 0.001]:
                    new_pts = pts.copy()
                    new_pts[i] = pts[i] + step * direction
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    r = _covering_radius(new_pts)
                    if r < best_r - 1e-10:
                        pts = new_pts
                        best_r = r
                        improved = True
                        break
        if not improved:
            break
    
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    np.random.seed(42)
    
    best_pts = None
    best_r = np.inf
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts1 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    result1 = _optimize_from_start(pts1, max_time=25.0)
    r1 = _covering_radius(result1)
    if r1 < best_r:
        best_r = r1
        best_pts = result1
    
    if _time_remaining() < 15.0:
        return best_pts
    
    # Start 2: Random
    pts2 = np.random.randn(N, 3)
    pts2 /= np.linalg.norm(pts2, axis=1, keepdims=True)
    
    result2 = _optimize_from_start(pts2, max_time=25.0)
    r2 = _covering_radius(result2)
    if r2 < best_r:
        best_r = r2
        best_pts = result2
    
    if _time_remaining() < 15.0:
        return best_pts
    
    # Start 3: Icosahedron-based (12 vertices, then add more)
    # Icosahedron vertices
    phi_gold = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi_gold, 0], [1, phi_gold, 0], [-1, -phi_gold, 0], [1, -phi_gold, 0],
        [0, -1, phi_gold], [0, 1, phi_gold], [0, -1, -phi_gold], [0, 1, -phi_gold],
        [phi_gold, 0, -1], [phi_gold, 0, 1], [-phi_gold, 0, -1], [-phi_gold, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Start from icosahedron and add random points, then relax
    pts3 = np.vstack([ico, np.random.randn(N - 12, 3)])
    pts3 /= np.linalg.norm(pts3, axis=1, keepdims=True)
    
    result3 = _optimize_from_start(pts3, max_time=25.0)
    r3 = _covering_radius(result3)
    if r3 < best_r:
        best_r = r3
        best_pts = result3
    
    if _time_remaining() < 10.0:
        return best_pts
    
    # Start 4: Another random
    np.random.seed(123)
    pts4 = np.random.randn(N, 3)
    pts4 /= np.linalg.norm(pts4, axis=1, keepdims=True)
    
    result4 = _optimize_from_start(pts4, max_time=25.0)
    r4 = _covering_radius(result4)
    if r4 < best_r:
        best_r = r4
        best_pts = result4
    
    return best_pts
# EVOLVE-BLOCK-END
