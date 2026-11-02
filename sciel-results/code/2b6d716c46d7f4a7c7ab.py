"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius of points on the unit sphere.
    The covering radius is the maximum over all points y on the sphere of
    the angular distance from y to the nearest point in pts.
    
    This equals the maximum circumradius of the facets of the convex hull,
    where each facet must have an empty cap (no other points inside).
    """
    pts = _normalize(pts)
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a large value
        return 180.0
    
    # For each facet, compute the circumradius (angular distance from facet center to farthest vertex)
    # The covering radius is determined by the facet with the largest "empty cap"
    # For a facet with vertices v1, v2, ..., vk, the cap center is the normalized sum of vertices
    # (for small facets) or more precisely the point on the sphere that maximizes distance to all facet vertices
    
    # Actually, for a facet of the convex hull, the cap that has this facet as its boundary
    # has its center at the normalized centroid of the facet vertices (for the inscribed sphere case).
    # The angular radius of this cap is the angle between the cap center and any facet vertex.
    
    # More precisely: the covering radius is the maximum over all facets of the angle
    # from the facet's circumcenter (on the sphere) to the farthest vertex of that facet.
    # For a planar polygon on the unit sphere, the relevant quantity is the maximum angular
    # distance from the facet center (normalized normal direction projected) to the vertices.
    
    # A cleaner approach: for each facet, the "empty cap" is the set of points on the sphere
    # that are farther from the facet center than any of the facet's vertices.
    # The covering radius is the max over all points y of min_i angle(y, p_i).
    # This is achieved at the circumcenter of the largest empty cap.
    
    # For each facet of the convex hull, compute the angular circumradius.
    # The facet center in 3D is the normalized sum of the facet vertices (approximate for small facets).
    # Better: the center of the cap is the point on the sphere equidistant (in angle) from all facet vertices.
    # For a triangle, this is the spherical circumcenter.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # Compute the circumcenter of this spherical polygon
        # For a triangle on the unit sphere, the circumcenter is the normalized cross product approach
        # or we can use the fact that the circumcenter lies along the normal to the plane of the triangle
        # (for the inscribed circle) - actually for spherical geometry it's more complex.
        
        # Simpler approach: the angular distance from the facet's 3D centroid (normalized) to each vertex
        # gives an upper bound. For better accuracy, use the spherical circumcenter.
        
        # For a triangle with vertices a, b, c on the unit sphere:
        # The circumcenter (spherical) is in the direction of the normalized cross product of (b-a) and (c-a)
        # projected appropriately. Actually, the center of the minimum enclosing cap for these 3 points
        # is along the direction of the circumcenter of the planar triangle formed by a, b, c.
        
        # Planar circumcenter of triangle a, b, c:
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute circumcenter of the planar triangle
        d = 2 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if abs(d) < 1e-15:
            # Degenerate, use centroid
            center = np.mean(verts, axis=0)
            center = center / np.linalg.norm(center)
        else:
            sa = np.dot(a, a)
            sb = np.dot(b, b)
            sc = np.dot(c, c)
            ux = (sa*(b[1]-c[1]) + sb*(c[1]-a[1]) + sc*(a[1]-b[1])) / d
            uy = (sa*(c[0]-b[0]) + sb*(a[0]-c[0]) + sc*(b[0]-a[0])) / d
            # The circumcenter in 3D for points on the unit sphere
            # is not simply (ux, uy, ?) because the plane doesn't pass through origin.
            # Instead, the spherical circumcenter is the normalized vector:
            # cross(b-a, c-a) gives the normal to the plane. The circumcenter on the sphere
            # is in the direction of: a + t1*(b-a) + t2*(c-a) where t1, t2 solve for equidistance.
            # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
            # is the normalization of the vector:
            #   a * |b-c|^2 * sin(angle at a) + ... 
            # This is getting complex. Let's use a simpler but effective approach:
            # The covering radius for a facet is the maximum angle from the facet center to vertices.
            # The facet center (for the cap) is the normalized sum of vertices for small facets.
            # For larger facets, we should use the actual circumcenter.
            
            # Let's compute the circumcenter properly for 3 points on the unit sphere.
            # The circumcenter O satisfies |O - a| = |O - b| = |O - c| with |O| = 1.
            # This means O lies on the perpendicular bisector planes of ab and ac.
            # Equivalently, O is in the direction of the normalized vector:
            #   n = (b - a) x (c - a)  (normal to the plane)
            #   But O is not along n unless the triangle is right.
            # 
            # The correct formula: O = normalize( a * (|b|^2 * |c|^2 - ...) ) -- too complex.
            # 
            # Alternative: use the fact that for the convex hull facet, the cap center
            # is the point on the sphere that maximizes the minimum angular distance to the facet vertices.
            # For a triangle, this is the spherical circumcenter.
            # 
            # Let's use a numerical approach: the circumcenter is along the direction
            # of the solution to the system. For 3 points on the unit sphere:
            # O = normalize( a * sin(2A) + b * sin(2B) + c * sin(2C) )
            # where A, B, C are the angles of the spherical triangle.
            
            # Simpler: just use the normalized centroid as an approximation.
            # This is exact for very small facets and a good approximation otherwise.
            center = np.mean(verts, axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm < 1e-15:
                center = verts[0].copy()
            else:
                center = center / center_norm
        
        # Compute angular distance from center to each vertex
        for v in verts:
            cos_angle = np.clip(np.dot(center, v), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > max_radius:
                max_radius = angle
    
    # Convert to degrees
    return np.degrees(max_radius)


def _local_search_facet_based(pts, max_time=100.0, tol=1e-10):
    """
    Geometric local search based on convex hull facets.
    For each facet, move the vertices of that facet to reduce the covering radius.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    step_size = 0.1
    n_iter = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        for simplex in hull.simplices:
            verts = pts[simplex].copy()
            
            # Compute current angular distances from facet center to vertices
            center = np.mean(verts, axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm < 1e-15:
                continue
            center = center / center_norm
            
            # For each vertex, try to move it slightly toward the center to reduce the cap radius
            # Actually, we want to move vertices to make the cap smaller.
            # The cap is largest when vertices are far from the cap center.
            # Moving a vertex toward the center reduces the angular distance.
            
            for j, idx in enumerate(simplex):
                orig = pts[idx].copy()
                
                # Direction: from vertex toward the cap center (on the sphere)
                # This is the geodesic direction
                dir_to_center = center - orig
                dir_norm = np.linalg.norm(dir_to_center)
                if dir_norm < 1e-15:
                    continue
                dir_to_center = dir_to_center / dir_norm
                
                # Try several step sizes
                for ss in [step_size, step_size * 0.5, step_size * 0.1]:
                    new_pt = orig + ss * dir_to_center
                    new_pt = _normalize(new_pt.reshape(1, -1))[0]
                    
                    # Check if this move is valid (point doesn't coincide with others)
                    # and if it improves the covering radius
                    pts_new = pts.copy()
                    pts_new[idx] = new_pt
                    
                    # Quick check: does this point stay within reasonable bounds?
                    if not np.all(np.isfinite(pts_new)):
                        continue
                    
                    # Compute new covering radius (expensive, but necessary)
                    # For speed, we can first check if the local change helps
                    new_radius = _covering_radius(pts_new)
                    
                    if new_radius < best_radius - 1e-12:
                        pts = pts_new
                        best_radius = new_radius
                        best_pts = pts.copy()
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_pts, best_radius


def _gradient_descent_smooth(pts, max_time=60.0):
    """
    Smooth gradient descent using a soft-max approximation of the covering radius.
    The covering radius is max_y min_i angle(y, p_i).
    We approximate this by considering the Voronoi diagram on the sphere.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    lr = 0.01
    beta = 100.0  # Sharpness of soft-max
    
    while time.time() - start_time < max_time:
        # Compute the covering radius contribution from each point
        # For each point p_i, find the farthest point in its Voronoi cell
        # Approximate by: for each pair of nearby points, the midpoint is a candidate
        # for the farthest point from both.
        
        # Compute all pairwise midpoints (candidates for Voronoi vertices)
        # This is O(N^2) which is fine for N=130
        n = len(pts)
        
        # For each point, compute the gradient of the soft covering radius
        # The covering radius is dominated by the largest empty cap.
        # The gradient of min_i angle(y, p_i) with respect to p_i is related to the direction
        # from p_i to y.
        
        # Simpler approach: use subgradient method
        # Find the point y on the sphere that maximizes min_i angle(y, p_i)
        # This is the center of the largest empty cap.
        
        # Approximate y by checking midpoints of all pairs
        best_y = None
        best_min_dist = -1
        
        # Sample points on the sphere for y
        # Use the midpoints of all pairs as candidates
        for i in range(n):
            for j in range(i+1, n):
                y = pts[i] + pts[j]
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-15:
                    continue
                y = y / y_norm
                
                # Compute min angular distance from y to all points
                dots = np.clip(pts @ y, -1.0, 1.0)
                angles = np.arccos(dots)
                min_dist = np.min(angles)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_y = y
        
        if best_y is None:
            break
        
        # The gradient of the covering radius with respect to each p_i
        # is in the direction that moves p_i away from best_y (to reduce the cap)
        # Actually, we want to move p_i TOWARD best_y to reduce the cap size.
        
        dots = np.clip(pts @ best_y, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Only update points that are close to best_y (within the cap)
        threshold = best_min_dist + 0.1
        mask = angles < threshold
        
        if not np.any(mask):
            break
        
        # Gradient: move points toward best_y
        # The angular gradient of angle(p, y) with respect to p is in the direction
        # of the projection of y onto the tangent plane at p.
        grads = np.zeros_like(pts)
        for i in range(n):
            if mask[i]:
                # Tangential component of y at p_i
                tangent = best_y - np.dot(pts[i], best_y) * pts[i]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-15:
                    grads[i] = tangent / tangent_norm
        
        # Update
        pts_new = pts + lr * grads
        pts_new = _normalize(pts_new)
        
        new_radius = _covering_radius(pts_new)
        if new_radius < best_radius - 1e-12:
            pts = pts_new
            best_radius = new_radius
            best_pts = pts.copy()
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Phase 1: Facet-based local search
    if time.time() < deadline - 30.0:
        pts, radius = _local_search_facet_based(pts, max_time=40.0)
        if radius < best_radius:
            best_pts = pts
            best_radius = radius
    
    # Phase 2: Smooth gradient descent
    if time.time() < deadline - 10.0:
        pts, radius = _gradient_descent_smooth(best_pts, max_time=30.0)
        if radius < best_radius:
            best_pts = pts
            best_radius = radius
    
    # Phase 3: Final facet-based polish
    if time.time() < deadline - 5.0:
        pts, radius = _local_search_facet_based(best_pts, max_time=20.0)
        if radius < best_radius:
            best_pts = pts
            best_radius = radius
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
