import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    return v / norms[:, np.newaxis]

def _compute_covering_radius_and_facets(points):
    """
    Compute convex hull, find facets, compute circumradii.
    Returns: covering_radius (max radius), facets (list of indices), radii (array)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., points coplanar or degenerate)
        return np.pi, [], np.array([])
    
    facets = hull.simplices
    radii = np.zeros(len(facets))
    
    for i, face in enumerate(facets):
        # face is a tuple of 3 indices
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        
        # Compute circumcenter of the spherical triangle
        # The circumcenter on the sphere is the normalized cross product of (p2-p1) and (p3-p1)? No.
        # For a triangle on a sphere, the circumcenter is in the plane of the triangle.
        # The normal to the plane is n = (p2-p1) x (p3-p1).
        # The circumcenter c satisfies |c - p1|^2 = |c - p2|^2 = |c - p3|^2.
        # Since c is on the sphere, |c|=1.
        # c lies on the line defined by the normal to the plane passing through the origin? No, the plane of the triangle does not pass through the origin unless the triangle is a great circle segment.
        # The circumcenter of a spherical triangle is the intersection of the sphere with the normal to the plane of the triangle? No.
        # The circumcenter of the *plane* triangle is the point equidistant from p1, p2, p3.
        # Let's find the circumcenter in 3D space first.
        
        # Vector approach:
        # Let c be the circumcenter.
        # c = p1 + u * (p2 - p1) + v * (p3 - p1) ? No, c is not necessarily in the affine span if we are on a sphere?
        # Actually, the circumcenter of the spherical triangle is the projection of the origin onto the plane defined by p1, p2, p3? No, that's the incenter of the polar triangle.
        
        # Correct method:
        # The circumcenter c is the unit vector in the direction of the normal to the plane of p1, p2, p3?
        # Let's check. If p1, p2, p3 are on the unit sphere, the plane equation is n . x = d.
        # n = (p2-p1) x (p3-p1). Normalize n.
        # d = n . p1.
        # The circumcenter is the point on the sphere closest to the plane? No.
        # The circumcenter is the point c such that c . p1 = c . p2 = c . p3 = cos(R).
        # This implies c is orthogonal to (p2-p1) and (p3-p1).
        # So c is parallel to (p2-p1) x (p3-p1).
        
        n = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[i] = np.pi # Degenerate
            continue
        c = n / norm_n
        # Ensure c points in the same direction as the facet normal (outward)
        # The hull normal is usually outward.
        # Check dot product with the centroid of the triangle
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        # The angular distance R is arccos(c . p1)
        dot_val = np.dot(c, p1)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    return max_radius, facets, radii

def construct():
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_and_facets(points)[0]
    
    # Lloyd's Algorithm iterations
    # Spherical Voronoi centroid approximation:
    # For each point, find adjacent triangles.
    # The centroid of the Voronoi cell is approximately the area-weighted average of the circumcenters of adjacent triangles.
    
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s for safety
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        # Compute hull and facets
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facets = hull.simplices
        num_facets = len(facets)
        
        # Map point index to list of adjacent facet indices
        point_facets = [[] for _ in range(N)]
        for idx, face in enumerate(facets):
            for p_idx in face:
                point_facets[p_idx].append(idx)
                
        new_points = points.copy()
        
        for i in range(N):
            if not point_facets[i]:
                continue
                
            # Sum of circumcenters weighted by area?
            # Area of spherical triangle is proportional to the solid angle.
            # Solid angle can be approximated by the area of the planar triangle or computed exactly.
            # Let's just use the circumcenters weighted by 1 for now, or by the edge lengths.
            # A better weight is the area of the triangle.
            
            sum_c = np.zeros(3)
            weight_sum = 0.0
            
            for f_idx in point_facets[i]:
                face = facets[f_idx]
                p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                
                # Compute circumcenter
                n = np.cross(p2 - p1, p3 - p1)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (p1 + p2 + p3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                    
                # Weight: area of triangle
                # Approximate area using vector cross product magnitude / 2
                area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
                
                sum_c += c * area
                weight_sum += area
                
            if weight_sum > 1e-10:
                new_centroid = sum_c / weight_sum
                new_points[i] = new_centroid
        
        # Normalize
        new_points = _normalize(new_points)
        
        # Evaluate
        new_radius, _, _ = _compute_covering_radius_and_facets(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
        else:
            # If it got worse, we might want to dampen the step
            # Or just stop if we are stuck?
            # Let's try a small step back
            pass
            
    # Local Polish: Gradient Descent on Max Radius
    # We want to minimize the max radius.
    # We can use a simple iterative scheme:
    # Find the facet with the largest radius.
    # Move its vertices slightly towards the circumcenter to reduce the radius?
    # Actually, moving vertices towards the circumcenter makes the triangle smaller, reducing the circumradius?
    # Let's test this logic.
    # If we move a vertex towards the circumcenter, the distance to the circumcenter decreases.
    # So the angular distance (radius) decreases.
    
    start_time_polish = time.time()
    deadline_polish = start_time_polish + 15.0
    
    step_size = 0.01
    for p_iter in range(500):
        if time.time() > deadline_polish:
            break
            
        radius, facets, radii = _compute_covering_radius_and_facets(points)
        
        if radii.size == 0:
            break
            
        max_idx = np.argmax(radii)
        if radii[max_idx] < 1e-6:
            break
            
        face = facets[max_idx]
        p1_idx, p2_idx, p3_idx = face
        
        # Compute circumcenter of this face
        p1, p2, p3 = points[p1_idx], points[p2_idx], points[p3_idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            break
        c = n / norm_n
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        # Move vertices towards c
        # The direction on the sphere from p to c is given by the geodesic.
        # Approximate: c - p, projected to tangent space at p.
        # Or simply: p + step * (c - p), then normalize.
        
        for idx in [p1_idx, p2_idx, p3_idx]:
            p = points[idx]
            # Project c onto tangent space at p
            # v = c - (p . c) p
            dot_pc = np.dot(p, c)
            v = c - dot_pc * p
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                v = v / norm_v
                points[idx] = points[idx] + step_size * v
                points[idx] = points[idx] / np.linalg.norm(points[idx])
                
        # Check improvement
        new_radius, _, _ = _compute_covering_radius_and_facets(points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = points.copy()
        else:
            # Revert
            points = best_points.copy()
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
                
    return best_points
