import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
_DEADLINE = 120.0

def _normalize(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        n = np.linalg.norm(pts)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / n
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1e-12
    pts = pts.copy()
    pts[mask] = pts[mask] / norms[mask, None]
    if np.any(~mask):
        pts[~mask] = np.array([1.0, 0.0, 0.0])
    return pts

def _compute_facet_metrics(pts, hull):
    """
    Vectorized computation of circumradii and circumcenters for all hull facets.
    Returns:
    radii: array of shape (n_facets,)
    centers: array of shape (n_facets, 3)
    faces: array of shape (n_facets, 3)
    """
    faces = hull.simplices
    p1 = pts[faces[:, 0]]
    p2 = pts[faces[:, 1]]
    p3 = pts[faces[:, 2]]
    
    # Circumcenter of spherical triangle p1, p2, p3
    # c is parallel to (p1 - p2) x (p2 - p3)
    v1 = p1 - p2
    v2 = p2 - p3
    normals = np.cross(v1, v2)
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Handle degenerate cases
    valid = norms[:, 0] > 1e-10
    centers = np.zeros_like(normals)
    centers[valid] = normals[valid] / norms[valid]
    
    # Ensure centers point towards the "inside" (same side as p1, p2, p3 relative to origin? 
    # Actually, for convex hull of points on sphere, the facet normal should point outward.
    # The circumcenter for the "empty cap" should be on the side of the facet facing away from the origin?
    # No, the circumcenter of the triangle formed by 3 points on the unit sphere is the point on the sphere 
    # equidistant to them. There are two such points (antipodal). We want the one that is the center of the 
    # empty cap, which is the one "outside" the convex hull? 
    # Wait, the convex hull facets are triangles. The "empty cap" is the region of the sphere not covered 
    # by the caps of radius r centered at the points. The center of the largest empty cap is the circumcenter 
    # of the Delaunay triangle. 
    # For points on the sphere, the circumcenter C satisfies C dot p1 = C dot p2 = C dot p3.
    # The vector (p1-p2)x(p2-p3) is normal to the plane of the triangle.
    # The circumcenter lies on this normal line.
    # We want the C such that C is on the unit sphere and maximizes the distance to the origin? No, C is ON the sphere.
    # So C = +/- normal / ||normal||.
    # Which sign? The one that is "farther" from the origin in terms of the convex hull? 
    # Actually, the circumcenter of the spherical triangle is the point on the sphere that is equidistant.
    # If the triangle is small, the circumcenter is close to the centroid.
    # The vector normal = (p1-p2)x(p2-p3) points in the direction of the cross product.
    # Let's check the dot product with p1. If C dot p1 < 0, it's the other side.
    # We want the circumcenter to be the one that is the "apex" of the empty cap.
    # Usually, for a convex hull of points on the sphere, the outward normal of the facet points away from the origin.
    # The circumcenter of the "empty" region (the hole) is the one that is NOT covered by the points.
    # If the points are the vertices, the "hole" is the region further from the vertices.
    # So C should be such that C is "outside" the triangle in a sense?
    # Let's just compute both and pick the one with larger dot product with the outward normal of the facet?
    # Or simpler: The circumcenter C is the solution to C dot p1 = C dot p2 = C dot p3 = cos(R).
    # Since C is on the sphere, C is a unit vector.
    # The vector n = (p1-p2)x(p2-p3) is orthogonal to p1-p2 and p2-p3.
    # So n dot p1 = n dot p2 = n dot p3.
    # So C = n/||n|| or -n/||n||.
    # cos(R) = C dot p1.
    # We want the circumradius R to be the distance from C to p1.
    # If we pick C such that C dot p1 is positive, R is in [0, pi/2].
    # If C dot p1 is negative, R is in [pi/2, pi].
    # The covering radius is typically small, so we expect R < pi/2.
    # So we should pick the sign such that C dot p1 > 0.
    
    dot_p1 = np.einsum('ij,ij->i', centers, p1)
    flip = dot_p1 < 0
    centers[flip] = -centers[flip]
    dot_p1[flip] = -dot_p1[flip]
    
    # Now dot_p1 is cos(R).
    cos_r = np.clip(dot_p1, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # Set invalid facets to 0 radius so they don't affect gradient
    radii[~valid] = 0.0
    centers[~valid] = 0.0
    
    return radii, centers, faces

def _local_optimize(pts, time_limit, max_iters=5000):
    """
    Vectorized gradient descent on all facets simultaneously.
    """
    pts = _normalize(pts).copy()
    start_time = time.time()
    best_pts = pts.copy()
    
    # Initial hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return best_pts
        
    radii, centers, faces = _compute_facet_metrics(pts, hull)
    best_radius = np.max(radii)
    
    lr = 0.05
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient
        # For each facet, if its radius is large, we want to move its vertices towards its circumcenter.
        # Weight the force by how much larger the radius is compared to the others, or just use the top K?
        # Using all facets with a weight proportional to (radius - mean) or just radius?
        # Let's use a soft-max style weighting to focus on the worst facets but still adjust others.
        # Or simpler: only move points that are part of facets with radius > current_best_radius * 0.95?
        # Let's try: force = sum over facets containing point i of (radius_f * direction)
        
        grad = np.zeros_like(pts)
        
        # We only apply force from facets that are "significant" (large radius)
        # To be efficient, let's just use all facets but scale by radius.
        # Actually, moving points towards the center of EVERY facet will shrink the whole mesh, 
        # which might not be what we want if the mesh is already uniform.
        # We want to expand the mesh in the directions of the large holes.
        # So, for a facet f with center c_f and radius r_f, the "force" on vertex v is (c_f - v) * r_f.
        # This pulls the vertex towards the center of the hole it helps define.
        
        # Vectorized accumulation
        # faces: (F, 3), radii: (F,), centers: (F, 3), pts: (N, 3)
        
        # For each facet f, and each vertex index j in 0..2:
        # grad[faces[f, j]] += radii[f] * (centers[f] - pts[faces[f, j]])
        
        # This is O(F * 3). F is ~ 2N - 4 = ~96. N=50. Very fast.
        
        for f in range(len(faces)):
            r = radii[f]
            if r < 1e-6: continue
            c = centers[f]
            for j in range(3):
                idx = faces[f, j]
                diff = c - pts[idx]
                # Normalize diff? No, the magnitude of diff is related to r.
                # Actually, moving towards c by a step proportional to r is good.
                grad[idx] += r * diff
                
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        valid_grad = grad_norm > 1e-12
        grad[valid_grad] = grad[valid_grad] / grad_norm[valid_grad]
        
        # Update
        new_pts = pts + lr * grad
        new_pts = _normalize(new_pts)
        
        # Evaluate new hull
        try:
            new_hull = ConvexHull(new_pts)
        except Exception:
            # If hull fails, revert
            pts = best_pts.copy()
            lr *= 0.9
            continue
            
        new_radii, new_centers, new_faces = _compute_facet_metrics(new_pts, new_hull)
        new_radius = np.max(new_radii)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            radii, centers, faces = new_radii, new_centers, new_faces
            # Keep lr or increase slightly?
            # lr = min(0.1, lr * 1.01)
        else:
            # Revert
            pts = best_pts.copy()
            radii, centers, faces = _compute_facet_metrics(pts, ConvexHull(pts))
            lr *= 0.95
            if lr < 0.001:
                break
                
    return best_pts

def construct():
    start_time = time.time()
    deadline = _DEADLINE - 5.0
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _compute_facet_metrics(pts, hull)
        best_radius = np.max(radii)
    except Exception:
        best_radius = 180.0
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts
    n_attempts = 10
    for attempt in range(n_attempts):
        if time.time() - start_time > deadline:
            break
            
        if attempt == 0:
            current = pts.copy()
        else:
            # Diverse initializations
            if rng.random() < 0.3:
                # Random
                current = _normalize(rng.normal(0, 1, size=(N, 3)))
            elif rng.random() < 0.5:
                # Perturbed best
                current = _normalize(best_pts + rng.normal(0, 0.5, size=(N, 3)))
            else:
                # Fibonacci with random rotation
                current = pts.copy()
                # Random rotation
                theta = rng.random() * 2 * np.pi
                phi_rot = rng.random() * np.pi
                c = np.cos(phi_rot)
                s = np.sin(phi_rot)
                R = np.array([
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                ])
                # Rotate around Z
                c_t = np.cos(theta)
                s_t = np.sin(theta)
                R_z = np.array([
                    [c_t, -s_t, 0],
                    [s_t, c_t, 0],
                    [0, 0, 1]
                ])
                current = current @ (R_z @ R).T
                current = _normalize(current)
                
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
            
        opt_pts = _local_optimize(current, time_limit=remaining - 5)
        
        try:
            opt_hull = ConvexHull(opt_pts)
            opt_radii, _, _ = _compute_facet_metrics(opt_pts, opt_hull)
            opt_radius = np.max(opt_radii)
        except Exception:
            opt_radius = 180.0
            
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
            
    # Final polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        final_pts = _local_optimize(best_pts, time_limit=remaining - 5)
        try:
            final_hull = ConvexHull(final_pts)
            final_radii, _, _ = _compute_facet_metrics(final_pts, final_hull)
            final_radius = np.max(final_radii)
        except Exception:
            final_radius = 180.0
            
        if final_radius < best_radius:
            best_pts = final_pts.copy()
            
    return _normalize(best_pts)
