"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _compute_covering_radius_and_grads(points):
    """
    Computes the covering radius and the gradient for each point.
    Returns: (covering_radius_rad, grads_array)
    """
    N = len(points)
    # Compute convex hull to get Delaunay triangulation on sphere
    # The facets of the convex hull of points on the unit sphere correspond to the Delaunay triangles.
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points, though unlikely on sphere)
        return np.pi, np.zeros((N, 3))
    
    # Facets are triangles (indices of vertices)
    # hull.simplices has shape (M, 3)
    simplices = hull.simplices
    
    # For each facet, compute the circumcenter and circumradius
    # The circumcenter of a spherical triangle (p1, p2, p3) is the normalized vector
    # in the direction of the normal to the plane containing p1, p2, p3, pointing away from the origin.
    # Actually, for points on the unit sphere, the circumcenter c is the normalized vector:
    # c = (p1 + p2 + p3) / |p1 + p2 + p3| is NOT generally correct for circumcenter.
    # The circumcenter is the intersection of the perpendicular bisectors.
    # In 3D, for a triangle on a sphere, the circumcenter is the normalized vector:
    # c = (p1 * |p2|^2 * |p3|^2 ... ) -- this is getting complex.
    
    # A robust way to find circumcenter of triangle on unit sphere:
    # The circumcenter c is equidistant from p1, p2, p3.
    # c . p1 = c . p2 = c . p3 = cos(R)
    # c is in the span of p1, p2, p3.
    # Let's solve for c = a*p1 + b*p2 + c*p3
    # c . p1 = cos(R), c . p2 = cos(R), c . p3 = cos(R)
    # This is a system of linear equations.
    
    # Alternatively, we can use the fact that the circumcenter is the normalized cross product of edge normals? No.
    # Let's use the formula for circumcenter of a triangle in 3D projected to sphere.
    # The plane containing p1, p2, p3 has normal n = (p2-p1) x (p3-p1).
    # The circumcenter lies on the line passing through the origin in the direction of n? No, that's the centroid of the face in the dual polytope.
    # Actually, for a spherical triangle, the circumcenter is the normalized vector:
    # c = (p1 * (p2.p3 - 1) + p2 * (p1.p3 - 1) + p3 * (p1.p2 - 1)) / norm(...) ? No.
    
    # Let's use the property that c is the solution to:
    # c . p1 = c . p2 = c . p3
    # and |c| = 1.
    # This implies c is orthogonal to (p1-p2) and (p1-p3).
    # So c is parallel to (p1-p2) x (p1-p3).
    # Let v = (p1-p2) x (p1-p3).
    # Then c = v / |v| or -v / |v|.
    # We need to choose the sign such that c is on the same side of the triangle as the "empty cap".
    # The empty cap condition means that no other point is closer to c than the vertices are.
    # Usually, the circumcenter for the Delaunay triangulation is the one that is "outside" the triangle relative to the origin?
    # No, the circumcenter of the spherical triangle is the point on the sphere equidistant from the vertices.
    # There are two such points (antipodal). The relevant one for the Voronoi cell is the one that is the center of the empty spherical cap.
    # The vector n = (p2-p1) x (p3-p1) points in the direction of the normal to the plane.
    # If we take c = normalize(n), does it give the circumcenter?
    # Let's check: n . (p1-p2) = 0 and n . (p1-p3) = 0. So n is orthogonal to the edges in the plane.
    # But we need c . p1 = c . p2 = c . p3.
    # If c = n / |n|, then c . p1 = c . p2 = c . p3 is NOT necessarily true.
    # Wait, the plane equation is n . x = d. Since p1, p2, p3 are on the unit sphere, d = n . p1 = n . p2 = n . p3.
    # So n is orthogonal to the chords, but not necessarily to the radial vectors.
    # However, the circumcenter of the triangle on the sphere is indeed the normalized vector n = (p2-p1) x (p3-p1) ?
    # Let's verify.
    # Let p1=(1,0,0), p2=(0,1,0), p3=(0,0,1).
    # p2-p1 = (-1,1,0), p3-p1 = (-1,0,1).
    # n = (1, 1, 1).
    # c = (1,1,1)/sqrt(3).
    # c . p1 = 1/sqrt(3). c . p2 = 1/sqrt(3). c . p3 = 1/sqrt(3).
    # Yes! It works. The circumcenter is the normalized cross product of the edge vectors.
    
    grads = np.zeros_like(points)
    max_circ_r = 0.0
    
    for simplex in simplices:
        i, j, k = simplex
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Edge vectors
        e1 = p2 - p1
        e2 = p3 - p1
        
        # Normal to the plane
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
            
        c = n / n_norm
        
        # Check if c is the "correct" circumcenter (empty cap).
        # The circumradius R is arccos(c . p1).
        # We need to ensure that for all other points p_l, c . p_l <= c . p1.
        # If not, we should use -c.
        # However, for Delaunay triangulation, the circumcenter should be the one where the cap is empty.
        # Let's check the dot products.
        dot1 = c @ p1
        # If dot1 is negative, it means the angle is > 90 deg, which is unlikely for small caps.
        # Actually, for the "empty cap" property, c should be closer to the vertices than to any other point.
        # This means c . p_l <= c . p1 for all l.
        # Let's compute min_dot for all points.
        dots = points @ c
        max_dot = np.max(dots)
        # The vertices should have the maximum dot product with c.
        # If max_dot > dot1 + 1e-6, then c is not the correct circumcenter for the empty cap.
        # In that case, use -c.
        if max_dot > dot1 + 1e-6:
            c = -c
            dot1 = -dot1
            
        # Circumradius
        cos_R = np.clip(dot1, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if R > max_circ_r:
            max_circ_r = R
            
        # Gradient contribution:
        # We want to move p1, p2, p3 to reduce R.
        # R = arccos(c . p1).
        # dR/dp1 = - (c / sqrt(1 - (c.p1)^2)) . (dc/dp1 + c . dp1) ?
        # Simpler: The gradient of the angular distance with respect to p1 on the sphere.
        # The direction to move p1 to decrease the distance to c is towards c.
        # But we must also consider that moving p1 changes c.
        # For a first-order approximation, we can ignore the change in c and just move p1 towards c.
        # The gradient of arccos(c . p) w.r.t p is -c / sqrt(1 - (c.p)^2) projected to tangent plane.
        # Let's compute the tangent component of c at p1.
        
        # Project c onto tangent plane at p1
        tan_c1 = c - (c @ p1) * p1
        # Gradient direction is opposite to this?
        # We want to minimize R. R increases as p1 moves away from c.
        # So we should move p1 towards c.
        # The vector pointing from p1 to c on the sphere is the tangent component of c.
        # So the update should be in the direction of tan_c1.
        # Wait, if we move p1 in the direction of tan_c1, the angle between p1 and c decreases.
        # So the gradient of R w.r.t p1 is in the direction of -tan_c1?
        # Let's check. f(p) = arccos(c . p).
        # df/dp = -c / sqrt(1 - (c.p)^2).
        # The projection of this gradient onto the tangent plane is what we need for Riemannian gradient.
        # proj_tangent(grad) = grad - (grad . p) p.
        # grad . p = -c . p / sqrt(1 - (c.p)^2).
        # So proj = -c / sqrt(...) + (c . p / sqrt(...)) p = (c . p p - c) / sqrt(...) = -tan_c1 / sqrt(...).
        # So the gradient points in the direction of -tan_c1.
        # To minimize R, we move in the direction of -gradient, which is tan_c1.
        # So we should add tan_c1 to the gradient (as a direction to move).
        
        # Normalize the contribution by the sensitivity (sqrt term)
        sin_R = np.sqrt(np.maximum(0.0, 1.0 - cos_R * cos_R))
        if sin_R < 1e-6:
            weight = 1.0
        else:
            weight = 1.0 / sin_R
            
        grads[i] += tan_c1 * weight
        grads[j] += (c - (c @ p2) * p2) * weight
        grads[k] += (c - (c @ p3) * p3) * weight
        
    return max_circ_r, grads

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = np.array([_normalize(p) for p in points])
    
    best_points = points.copy()
    best_score = _compute_covering_radius_and_grads(best_points)[0]
    
    lr = 0.05
    n_iter = 2000
    repulsion_strength = 0.1
    min_dist = 0.05  # Radians, ~3 degrees
    
    for iteration in range(n_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute covering radius and gradients
        cr, grads = _compute_covering_radius_and_grads(points)
        
        # Add repulsion force to prevent points from collapsing
        # For each pair of points closer than min_dist, add a repulsive force
        # This is O(N^2) but N=130 is small.
        for i in range(N):
            for j in range(i + 1, N):
                d = np.arccos(np.clip(points[i] @ points[j], -1.0, 1.0))
                if d < min_dist:
                    # Repulsion
                    # Force is proportional to (min_dist - d)
                    force_mag = repulsion_strength * (min_dist - d)
                    # Direction: i away from j, j away from i
                    # Tangent direction from j to i
                    tan_ij = points[i] - (points[i] @ points[j]) * points[j]
                    tan_ij = _normalize(tan_ij) if np.linalg.norm(tan_ij) > 1e-12 else np.zeros(3)
                    grads[i] += tan_ij * force_mag
                    grads[j] -= tan_ij * force_mag
                    
        # Normalize gradients
        grad_norms = np.linalg.norm(grads, axis=1)
        max_grad = np.max(grad_norms) if len(grad_norms) > 0 else 1.0
        if max_grad > 1e-12:
            grads = grads / max_grad
        
        # Riemannian gradient step
        for idx in range(N):
            p = points[idx]
            g = grads[idx]
            new_p = p + lr * g
            new_p = _normalize(new_p)
            points[idx] = new_p
            
        # Evaluate
        if iteration % 10 == 0 or iteration == n_iter - 1:
            score, _ = _compute_covering_radius_and_grads(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()
                
        # Adaptive learning rate
        if iteration % 100 == 0 and iteration > 0:
            lr *= 0.95
            
    return best_points
# EVOLVE-BLOCK-END
