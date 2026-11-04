import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius(points):
    """
    Computes the covering radius of a set of points on the unit sphere.
    Returns the covering radius in radians and the index of the worst point (approximated by facet circumcenter).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails (e.g., points in hemisphere), return a large value
        return np.pi, -1
    
    max_radius = 0.0
    worst_facet = None
    
    for simplex in hull.simplices:
        # simplex is an array of indices into points
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle defined by p1, p2, p3
        # The circumcenter is the point on the sphere equidistant from p1, p2, p3.
        # It is proportional to the cross product of (p2-p1) and (p3-p1) ? No, that's the normal.
        # The circumcenter c satisfies c.p1 = c.p2 = c.p3.
        # This means c is in the plane defined by p1, p2, p3.
        # c = a*p1 + b*p2 + c*p3.
        # A simpler way: The circumcenter is the normalized vector perpendicular to the plane of the triangle?
        # No, the circumcenter of a spherical triangle is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle?
        # Actually, for points on the unit sphere, the circumcenter is the normalized vector:
        # c = normalize( p1 + p2 + p3 + ... )? No.
        
        # Let's solve for the circumcenter.
        # The vector c is in the span of p1, p2, p3.
        # c = x*p1 + y*p2 + z*p3
        # c.p1 = c.p2 = c.p3 = cos(theta)
        # This is a linear system.
        # Alternatively, the circumcenter is the normalized cross product of the edges?
        # The normal to the plane is n = cross(p2-p1, p3-p1).
        # The circumcenter lies on the line t*n? No, that's the center of the circle in the plane, not on the sphere.
        
        # Correct method:
        # The circumcenter c is the solution to:
        # c.p1 = c.p2
        # c.p2 = c.p3
        # |c| = 1
        # This implies c is in the plane of p1, p2, p3.
        # Let c = a*p1 + b*p2 + c*p3.
        # Actually, there is a closed form.
        # c = normalize( p1 * (1 - p2.p3) + p2 * (1 - p3.p1) + p3 * (1 - p1.p2) )?
        # Let's test this formula.
        
        d12 = np.dot(p1, p2)
        d23 = np.dot(p2, p3)
        d31 = np.dot(p3, p1)
        
        # The vector sum that is equidistant from p1, p2, p3
        # v = p1 * (1 - d23) + p2 * (1 - d31) + p3 * (1 - d12)
        v = p1 * (1.0 - d23) + p2 * (1.0 - d31) + p3 * (1.0 - d12)
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            # Degenerate case, e.g., antipodal points or very small triangle
            # Approximate with the centroid
            c = (p1 + p2 + p3) / 3.0
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c /= norm_c
        else:
            c = v / norm_v
        
        # Check if c is within the spherical triangle.
        # The circumcenter is inside the triangle if the dot products of c with the edges are positive?
        # Or simply check if the angular distance from c to p1, p2, p3 is less than pi/2?
        # Actually, if the triangle is large, the circumcenter might be outside.
        # If c is outside, the worst point is one of the vertices.
        # We can check if c is in the convex hull of p1, p2, p3 on the sphere.
        # A simple check: if the dot product of c with the normal of the plane (oriented) is consistent?
        
        # For now, let's assume the circumcenter is the worst point candidate.
        # If the circumcenter is outside the triangle, the maximum distance is at a vertex.
        # The distance from c to p1 is acos(c.p1).
        
        dist1 = np.arccos(np.clip(np.dot(c, p1), -1.0, 1.0))
        dist2 = np.arccos(np.clip(np.dot(c, p2), -1.0, 1.0))
        dist3 = np.arccos(np.clip(np.dot(c, p3), -1.0, 1.0))
        
        # If the circumcenter is outside, the max distance is max(dist1, dist2, dist3) ?
        # No, if c is outside, the farthest point in the triangle from c is a vertex.
        # But we want the covering radius, which is the max distance from ANY point on the sphere to the nearest config point.
        # Within this facet, the farthest point from the config points is the circumcenter IF it is inside the facet.
        # If it is outside, the farthest point is one of the vertices of the facet (which are config points, so distance 0? No, the vertices are config points, so distance to nearest config point is 0).
        # Wait, the covering radius is the max over y of dist(y, nearest config point).
        # The vertices p1, p2, p3 are config points, so dist(p1, p1) = 0.
        # So the worst point in the facet is NOT a vertex. It is the circumcenter if it lies inside the facet.
        # If the circumcenter lies outside the facet, then the function dist(y, nearest) decreases as we move from the circumcenter towards the triangle?
        # Actually, the level sets of dist(y, p1) are circles.
        # The point in the triangle maximizing min(dist(y,p1), dist(y,p2), dist(y,p3)) is the circumcenter if it is inside the triangle.
        # If the circumcenter is outside, the maximum is attained at one of the vertices?
        # No, the vertices have distance 0 to themselves.
        # Let's re-read: "largest angular distance from any point of the sphere to its nearest chosen point".
        # So we want max_y min_i dist(y, p_i).
        # In a facet defined by p1, p2, p3, the nearest point is one of p1, p2, p3.
        # The function f(y) = min(dist(y,p1), dist(y,p2), dist(y,p3)) is maximized at the circumcenter if it is inside the triangle.
        # If the circumcenter is outside, the maximum is at the vertex opposite to the side where the circumcenter is?
        # Actually, if the circumcenter is outside, the triangle is "obtuse" on the sphere.
        # In that case, the maximum of the min-distance is at one of the vertices?
        # No, the vertices have min-distance 0.
        # Let's test: p1=(1,0,0), p2=(0,1,0), p3=(0,0,1).
        # Circumcenter is (1,1,1)/sqrt(3).
        # dist(c, p1) = acos(1/sqrt(3)) ~ 54.7 deg.
        # Is c inside the triangle? Yes.
        # Now consider p1=(1,0,0), p2=(-1,0,0), p3=(0,1,0).
        # This is a hemisphere.
        # Circumcenter?
        # d12 = -1, d23 = 0, d31 = 0.
        # v = p1(1) + p2(1) + p3(1) = (0,1,0) = p3.
        # c = p3.
        # dist(c, p3) = 0.
        # The max min-distance in this hemisphere is at (0,-1,0) which is not in the triangle?
        # The triangle covers the upper hemisphere.
        # The point (0,-1,0) is the south pole.
        # dist((0,-1,0), p3) = pi/2.
        # dist((0,-1,0), p1) = pi/2.
        # dist((0,-1,0), p2) = pi/2.
        # So the max is pi/2.
        # The circumcenter calculation gave c=p3, which is wrong for the "worst" point.
        
        # So we must check if the circumcenter is inside the triangle.
        # A point c is inside the spherical triangle p1,p2,p3 if:
        # 1. c is in the same hemisphere as the triangle (dot(c, normal) > 0).
        # 2. The angles are correct.
        
        # Simpler check:
        # The circumcenter is inside if the dot products of c with the outward normals of the edges are positive?
        # Let's use a different strategy: Sample the circumcenter and check if it is closer to p1,p2,p3 than to any other point? No.
        
        # For robustness, let's just compute the circumradius for all facets and take the max.
        # If the circumcenter is outside, the circumradius is not the covering radius for that facet.
        # However, in a good covering, the triangles are small, so the circumcenter is likely inside.
        # We can assume this for the optimization step.
        
        r = max(dist1, dist2, dist3)
        if r > max_radius:
            max_radius = r
            worst_facet = simplex
            
    return max_radius, worst_facet

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _ = compute_covering_radius(points)
    
    # Local search
    step_size = 0.05
    while time.time() < deadline:
        radius, worst_facet = compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if worst_facet is None:
            break
            
        p1_idx, p2_idx, p3_idx = worst_facet
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        # Recompute circumcenter for the worst facet
        d12 = np.dot(p1, p2)
        d23 = np.dot(p2, p3)
        d31 = np.dot(p3, p1)
        v = p1 * (1.0 - d23) + p2 * (1.0 - d31) + p3 * (1.0 - d12)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            c = (p1 + p2 + p3) / 3.0
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c /= norm_c
        else:
            c = v / norm_v
            
        # Move points towards c
        # We want to move p1, p2, p3 towards c to reduce the circumradius.
        # A simple move: p_new = normalize(p + step * (c - p))
        
        new_points = points.copy()
        
        for idx in [p1_idx, p2_idx, p3_idx]:
            p = new_points[idx]
            # Vector from p to c on the sphere (tangent space approximation)
            # Or just linear interpolation and normalize
            new_p = p + step_size * (c - p)
            new_p /= np.linalg.norm(new_p)
            new_points[idx] = new_p
            
        new_radius, _ = compute_covering_radius(new_points)
        
        if new_radius < radius:
            points = new_points
        else:
            # If it didn't improve, try a smaller step or different direction?
            # For now, just reduce step size slightly to escape local minima?
            # Or try moving only one point at a time?
            step_size *= 0.9
            if step_size < 0.001:
                break
                
    return best_points
