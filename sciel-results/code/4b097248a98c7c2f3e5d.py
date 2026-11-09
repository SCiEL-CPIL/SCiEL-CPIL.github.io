"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
import scipy.optimize as opt

def _initial_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _cover_radius(pts):
    """
    Compute the covering radius.
    The covering radius is the maximum over the sphere of the distance to the nearest point.
    For a set of points on the sphere, the farthest point from the set is the circumcenter
    of one of the facets of the convex hull that contains the origin in its normal cone
    (i.e., the facet is visible from the origin, or rather, the origin is on the 'inside' side).
    Actually, for points on the unit sphere, the convex hull contains the origin if the points
    are not in a hemisphere. The covering radius is the max distance from the origin to the
    planes of the facets that "face away" from the origin?
    No. The covering radius is the max distance from any point on the sphere to the set.
    This is equal to the max circumradius of the spherical Delaunay triangles.
    The circumcenter of a spherical triangle (p1, p2, p3) is the point c such that
    c is equidistant from p1, p2, p3.
    In 3D, this is the intersection of the sphere with the plane bisecting the triangle?
    No, it's the center of the smallest circle containing p1, p2, p3 on the sphere.
    This corresponds to the circumcenter of the planar triangle formed by p1, p2, p3,
    IF the origin is inside the tetrahedron formed by the origin and the triangle?
    Actually, the circumcenter of the spherical triangle is the point on the sphere
    that is the normalized intersection of the line from the origin through the
    circumcenter of the planar triangle (p1, p2, p3).
    Let C be the circumcenter of the planar triangle p1, p2, p3.
    Then the spherical circumcenter is C / ||C||.
    The distance is arccos(dot(C/||C||, p1)).
    We need to consider only the "empty" circles, i.e., the facets of the convex hull
    of the points that contain the origin in their normal cone?
    Wait, the covering radius is determined by the largest empty circle.
    The centers of the largest empty circles are the circumcenters of the Delaunay triangles
    (spherical).
    The Delaunay triangulation on the sphere is dual to the convex hull.
    Specifically, a triangle (i, j, k) is a Delaunay triangle if and only if the facet
    (i, j, k) of the convex hull is such that the origin is on the "inside" of the plane
    defined by i, j, k? No.
    The origin is inside the convex hull.
    A facet (i, j, k) is part of the convex hull.
    The normal vector n to the facet points away from the origin (since origin is inside).
    The circumcenter of the spherical triangle is in the direction of n?
    Let's check. If the origin is inside, the plane ax+by+cz=d with d>0 (since origin is at 0 and normal points out).
    The circumcenter of the planar triangle is on the line from the origin perpendicular to the plane?
    No.
    However, it is a known fact that for points on the sphere, the circumcenters of the
    Delaunay triangles (which correspond to the facets of the convex hull) are the candidates
    for the center of the covering circle.
    The distance is the angle between the circumcenter and the vertices.
    We should compute the circumcenter for every facet of the convex hull.
    Then the covering radius is the maximum of these distances.
    Note: If the circumcenter of the planar triangle is on the opposite side of the plane
    from the origin, it might not be the "spherical" circumcenter in the sense of the
    smaller arc?
    Actually, for the convex hull of points on the sphere, the origin is inside.
    The plane of the facet does not separate the origin from the points?
    The origin is on one side, the rest of the hull on the other? No, the hull is convex.
    The origin is inside. So the plane of the facet has the origin on the "inner" side.
    The normal vector of the facet (pointing outwards) has a positive dot product with
    the circumcenter of the planar triangle?
    Let's just compute the circumcenter of the planar triangle (p1, p2, p3).
    Let c be the circumcenter.
    The distance from the origin to the plane is h.
    The distance from c to the origin is r_c.
    The circumradius of the spherical triangle is arccos( (h^2 + r_c^2 - 1) / (2 * r_c * h) )? No.
    Simply: The spherical circumcenter is c / ||c||.
    The distance is arccos( dot(c/||c||, p1) ).
    This works if c is not zero.
    If c is zero (points are orthogonal?), the circle is a hemisphere?
    
    Algorithm:
    1. Compute ConvexHull of pts.
    2. For each simplex (facet) in hull.simplices:
       a. Get p1, p2, p3.
       b. Compute circumcenter c of triangle p1, p2, p3.
       c. If ||c|| is small, skip? Or handle carefully.
       d. Compute angle = arccos( dot(c/||c||, p1) ).
       e. Keep max angle.
    
    Wait, is it possible that the covering center is not a circumcenter of a hull facet?
    The covering radius is the max over y of min_i dist(y, p_i).
    The function f(y) = min_i dist(y, p_i) is concave? No, min of concave is not necessarily concave.
    But the "Voronoi" cells on the sphere are convex (spherically).
    The maximum of the distance to the boundary of the Voronoi cell is at a vertex of the cell.
    The vertices of the Voronoi cells are the circumcenters of the Delaunay triangles.
    So yes, checking the circumcenters of the Delaunay triangles is sufficient.
    The Delaunay triangles on the sphere correspond to the facets of the convex hull.
    (Specifically, the dual graph).
    So we just need to iterate over hull.simplices.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (e.g., coplanar, though unlikely on sphere)
        # Use a grid search as a fallback
        return _cover_radius_grid(pts)
    
    max_rad = 0.0
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        # Circumcenter of triangle p1, p2, p3
        # Solve |c - p1|^2 = |c - p2|^2 = |c - p3|^2 = R^2
        # 2 c . (p2 - p1) = |p2|^2 - |p1|^2 = 0 (since on unit sphere)
        # 2 c . (p3 - p1) = 0
        # So c is orthogonal to (p2 - p1) and (p3 - p1).
        # Thus c is parallel to (p2 - p1) x (p3 - p1).
        v1 = p2 - p1
        v2 = p3 - p1
        c = np.cross(v1, v2)
        
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            continue
        
        c = c / norm_c
        
        # The circumcenter could be c or -c.
        # We want the one that is "inside" the triangle on the sphere?
        # Actually, for the convex hull facet, the normal points outwards.
        # The origin is inside.
        # The circumcenter of the spherical triangle (the one with empty cap)
        # should be on the same side of the plane as the origin?
        # No, the circumcenter is the center of the circle passing through p1, p2, p3.
        # There are two such points on the sphere: c and -c.
        # One corresponds to the triangle, the other to the "complement" triangle.
        # We want the one that is the center of the empty cap.
        # The empty cap is the one that doesn't contain other points.
        # For a convex hull facet, the "outside" is empty.
        # The normal vector n = c (normalized) points to one of the poles.
        # If we choose c such that it is on the same side of the plane as the origin?
        # The plane equation: n . x = d.
        # Since p1 is on the plane, n . p1 = d.
        # Origin is at 0. n . 0 = 0.
        # Since origin is inside the hull, and the facet is on the boundary,
        # the origin and the interior of the hull are on the side where n . x < d?
        # Or n . x > d?
        # The normal points outwards. So for any point x in the hull, n . x <= d.
        # So n . 0 = 0 <= d. Thus d >= 0.
        # The circumcenter c is at distance R from the origin? No, c is on the unit sphere.
        # The distance from origin to plane is d = n . p1.
        # The circumradius of the planar triangle is r_planar.
        # r_planar^2 = ||c - p1||^2 = ||c||^2 + ||p1||^2 - 2 c . p1 = 2 - 2 c . p1.
        # Also r_planar^2 = d^2 + (distance from projection of c to plane?)
        # This is getting complicated.
        
        # Simpler check:
        # The spherical circumcenter is the point on the sphere equidistant from p1, p2, p3.
        # It is either c or -c.
        # The distance to p1 is arccos(c . p1) or arccos(-c . p1).
        # One is small, one is large (sum to pi).
        # The "empty" circle is the one that contains no other points in its interior.
        # For a facet of the convex hull, the "exterior" side is empty.
        # The center of the empty circle on the exterior side is the one that is
        # "visible" from outside.
        # This corresponds to the normal vector pointing outwards.
        # So we should check if c . n_hull > 0?
        # We don't have n_hull explicitly, but c is parallel to the normal.
        # If c . p1 > 0, then c is "close" to p1.
        # If c . p1 < 0, c is "far" from p1.
        # The circumcenter of the triangle on the sphere (the one with the smaller cap)
        # is the one closer to the vertices.
        # So we should pick the sign such that c . p1 is positive?
        # Let's assume the circumcenter of the "small" triangle is the one with c . p1 > 0.
        # If c . p1 < 0, we flip c to -c.
        
        if c[0]*p1[0] + c[1]*p1[1] + c[2]*p1[2] < 0:
            c = -c
            
        dot_val = np.dot(c, p1)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        rad = np.arccos(dot_val)
        
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

def _cover_radius_grid(pts):
    # Fallback: grid search
    # This is slow, so only use if hull fails
    max_rad = 0.0
    # Coarse grid
    lat = np.linspace(-np.pi/2, np.pi/2, 50)
    lon = np.linspace(0, 2*np.pi, 100)
    L, O = np.meshgrid(lat, lon)
    x = np.cos(L) * np.cos(O)
    y = np.cos(L) * np.sin(O)
    z = np.sin(L)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute min distance for each grid point
    # This is O(G * N). G=5000, N=130. 650k ops. Fast enough.
    for gp in grid_pts:
        dists = np.arccos(np.clip(gp @ pts.T, -1.0, 1.0))
        min_d = np.min(dists)
        if min_d > max_rad:
            max_rad = min_d
    return max_rad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize
    pts = _initial_fibonacci(N)
    
    best_pts = pts.copy()
    best_rad = _cover_radius(best_pts)
    
    # 2. Optimization Loop
    # We will use a custom gradient descent / local search.
    # Objective: minimize covering radius.
    # Since _cover_radius is non-differentiable (max of arccos),
    # we can use a subgradient method or just perturb and accept if better.
    
    # To make it faster and more robust, we can use a "repulsive" force model
    # that approximates the covering radius reduction.
    # Force on point i:
    # If point i is the nearest neighbor to the "worst" point (circumcenter),
    # it should move away? No, the worst point is far from all points.
    # Moving points towards the worst point reduces the distance.
    # But we must maintain the "even" distribution.
    
    # Let's try a simple stochastic local search with a twist:
    # Identify the current "worst" circumcenter (hole).
    # Move the 3 points forming that hole slightly towards the hole?
    # Or move them apart to "open" the hole? No, moving them towards the hole
    # shrinks the hole.
    # But moving them towards the hole might create a new hole elsewhere.
    
    # Let's implement a gradient step based on the active constraints.
    # The active constraints are the vertices of the "worst" Voronoi cell.
    # Let c be the worst circumcenter.
    # Let i, j, k be the indices of the points forming the triangle.
    # The distance d = arccos(c . p_i).
    # We want to minimize d.
    # Gradient of d w.r.t p_i:
    # d = arccos(u), u = c . p_i.
    # dd/du = -1 / sqrt(1 - u^2).
    # du/dp_i = c.
    # So grad_p_i d = -c / sqrt(1 - (c.p_i)^2).
    # We want to decrease d, so we move p_i in the direction of -grad = c / sqrt(...).
    # So we move p_i towards c.
    
    # Let's try this:
    # 1. Find worst circumcenter c and its triangle (i, j, k).
    # 2. Compute gradients for p_i, p_j, p_k.
    # 3. Take a small step in that direction.
    # 4. Normalize.
    # 5. Repeat.
    
    # We need to find the worst circumcenter and its triangle efficiently.
    
    def get_worst_hole(pts):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None
            
        max_rad = 0.0
        worst_c = None
        worst_simplex = None
        
        for simplex in hull.simplices:
            p1 = pts[simplex[0]]
            p2 = pts[simplex[1]]
            p3 = pts[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            c = np.cross(v1, v2)
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-12:
                continue
            c = c / norm_c
            
            if np.dot(c, p1) < 0:
                c = -c
                
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            rad = np.arccos(dot_val)
            
            if rad > max_rad:
                max_rad = rad
                worst_c = c
                worst_simplex = simplex
                
        return worst_c, worst_simplex, max_rad

    # Optimization
    step_size = 0.01
    iterations = 0
    max_iter = 5000
    
    while time.time() < deadline and iterations < max_iter:
        iterations += 1
        
        c, simplex, rad = get_worst_hole(pts)
        if c is None or simplex is None:
            break
            
        # If the radius is very small, we can stop or reduce step size
        # if rad < 0.19: # ~11 degrees
        #     break
        
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        # Compute gradients
        # grad_i = c / sqrt(1 - (c.p_i)^2)
        # Note: c.p_i is the cosine of the angle, which is < 1.
        # To avoid division by zero, clip the denominator.
        
        def grad(p):
            dot_val = np.dot(c, p)
            denom = np.sqrt(np.maximum(1e-6, 1.0 - dot_val * dot_val))
            return c / denom
        
        g1 = grad(p1)
        g2 = grad(p2)
        g3 = grad(p3)
        
        # Normalize gradients to have unit length?
        # The magnitude of the gradient depends on the angle.
        # Steeper gradient near the equator of the cap?
        # Let's just normalize to unit vectors to keep step size consistent.
        g1 = g1 / np.linalg.norm(g1)
        g2 = g2 / np.linalg.norm(g2)
        g3 = g3 / np.linalg.norm(g3)
        
        # Update points
        # We move p1, p2, p3 towards c.
        # But we must be careful not to move them too far.
        # Also, we should consider the effect on other points.
        # A simple approach: move only the 3 points.
        
        pts[simplex[0]] = p1 + step_size * g1
        pts[simplex[1]] = p2 + step_size * g2
        pts[simplex[2]] = p3 + step_size * g3
        
        # Normalize
        pts = _normalize(pts)
        
        # Check if improved
        new_rad = _cover_radius(pts)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = pts.copy()
            # Optionally increase step size slightly?
            # step_size = min(0.05, step_size * 1.05)
        else:
            # Revert
            pts[simplex[0]] = p1
            pts[simplex[1]] = p2
            pts[simplex[2]] = p3
            pts = _normalize(pts)
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
        
        # Every 100 iterations, do a full check and maybe a global shuffle?
        # For now, just continue.
        
    # Final Polish
    # Use a smaller step size and more iterations if time permits
    if time.time() < deadline - 5.0:
        step_size = 1e-4
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            c, simplex, rad = get_worst_hole(pts)
            if c is None:
                break
            p1 = pts[simplex[0]]
            p2 = pts[simplex[1]]
            p3 = pts[simplex[2]]
            
            def grad(p):
                dot_val = np.dot(c, p)
                denom = np.sqrt(np.maximum(1e-6, 1.0 - dot_val * dot_val))
                return c / denom
            
            g1 = grad(p1) / np.linalg.norm(grad(p1))
            g2 = grad(p2) / np.linalg.norm(grad(p2))
            g3 = grad(p3) / np.linalg.norm(grad(p3))
            
            pts[simplex[0]] = p1 + step_size * g1
            pts[simplex[1]] = p2 + step_size * g2
            pts[simplex[2]] = p3 + step_size * g3
            pts = _normalize(pts)
            
            new_rad = _cover_radius(pts)
            if new_rad < best_rad:
                best_rad = new_rad
                best_pts = pts.copy()
            else:
                pts[simplex[0]] = p1
                pts[simplex[1]] = p2
                pts[simplex[2]] = p3
                pts = _normalize(pts)
                
    return best_pts
# EVOLVE-BLOCK-END
