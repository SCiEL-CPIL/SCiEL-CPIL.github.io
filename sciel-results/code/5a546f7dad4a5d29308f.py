"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _covering_radius(points):
    """
    Calculate the covering radius of the configuration.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest neighbor in the configuration.
    This is equivalent to the maximum circumradius of the Delaunay triangulation facets.
    """
    points = _normalize(points)
    if len(points) < 4:
        # Degenerate case
        return np.pi / 2.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fallback to a rough estimate or return a large value
        # This can happen if points are coplanar or too few, but N=50 should be fine
        return np.pi / 2.0
        
    facets = hull.simplices
    
    max_radius = 0.0
    for facet in facets:
        p1 = points[facet[0]]
        p2 = points[facet[1]]
        p3 = points[facet[2]]
        
        # Compute angular distances
        d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
        d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
        d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
        
        # Check if the spherical triangle is acute.
        # If acute, the circumcenter is inside, and the radius is the circumradius.
        # If obtuse, the maximum distance is at the midpoint of the longest side.
        
        # A spherical triangle is acute if all its angles are < 90 degrees.
        # Alternatively, we can check if the circumcenter lies within the triangle.
        # A simpler check: if the longest side squared > sum of squares of other two (Euclidean approximation for small angles)
        # But for large angles, we should use spherical geometry.
        
        # Let's compute the circumcenter.
        # The circumcenter c is equidistant from p1, p2, p3.
        # It lies on the normal to the plane of the triangle.
        # c = k * (p1 x p2 + p2 x p3 + p3 x p1) ? No.
        # The circumcenter of a triangle on the unit sphere can be found by normalizing the vector sum of the cross products of the sides?
        # Actually, the circumcenter is the normalization of (p1 x p2) x (p2 x p3) ... no.
        
        # The plane equation is n . x = d.
        # n = (p1 - p2) x (p1 - p3)
        # The circumcenter lies on the line from the origin through the circumcenter of the Euclidean triangle? No.
        
        # Let's use the property that the circumcenter c satisfies:
        # |c - p1|^2 = |c - p2|^2 = |c - p3|^2
        # Since |c|=1, |p_i|=1, this implies c . p1 = c . p2 = c . p3.
        # So c is orthogonal to (p1 - p2) and (p1 - p3).
        # c is parallel to (p1 - p2) x (p1 - p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_vec = np.cross(v1, v2)
        
        c_norm = np.linalg.norm(c_vec)
        if c_norm < 1e-10:
            # Points are collinear or coincident
            max_side = max(d12, d23, d31)
            current_radius = max_side / 2.0
        else:
            c = c_vec / c_norm
            # Check if c is on the same side of the edges as the interior of the triangle.
            # The circumcenter is inside the spherical triangle if the dot product of c with each vertex is greater than the dot product of the circumcenter with the opposite side's midpoint?
            # Simpler: The circumcenter is inside if the triangle is acute.
            # For a spherical triangle, it is acute if the circumcenter is inside.
            # We can check if c is a convex combination of p1, p2, p3? No, it's on the sphere.
            # Check if the angles at the vertices are < 90.
            # Angle at p1: cos(A) = ( (p2-p1). (p3-p1) ) / ( |p2-p1| |p3-p1| )? No, that's Euclidean.
            # Spherical angle: cos(A) = (sin(c) - sin(b)cos(a)) / (sin(a)sin(b))? Too complex.
            
            # Alternative check:
            # The circumcenter c is inside the triangle if the dot product of (c - p1) with the normal to the edge (p1,p2) pointing inward is positive?
            # Actually, if the triangle is obtuse, the circumcenter lies outside.
            # We can determine if the circumcenter is inside by checking if the dot product of c with each vertex is greater than the dot product of the circumcenter of the *antipodal* triangle?
            
            # Let's use a robust check:
            # If the circumradius R_c = arccos(c . p1) is less than the half-perimeter? No.
            # If the triangle is acute, R_c < 90 degrees? No, R_c can be anything.
            
            # Let's just check if the circumcenter lies in the convex hull of the 3 points in 3D? No, it's on the sphere.
            # The circumcenter is inside the spherical triangle if and only if the triangle is acute.
            # We can check if the dot product of the normal to the face (pointing out) with c is positive?
            # The normal n = (p1-p2) x (p1-p3) points outwards if the orientation is correct.
            # If n . c > 0, c is on the same side as the normal.
            
            # Let's compute the circumradius
            cos_rc = np.clip(np.dot(c, p1), -1.0, 1.0)
            rc = np.arccos(cos_rc)
            
            # Check if c is inside the triangle.
            # If the triangle is obtuse, the maximum distance is at the midpoint of the longest side.
            # How to check if it's obtuse?
            # In a spherical triangle, if one angle is > 90, it's obtuse.
            # Angle at p1: 
            # u = p2 - p1, v = p3 - p1
            # cos_angle = (u . v) / (|u| |v|)
            # This is the Euclidean angle between the chords. It's related to the spherical angle.
            # Actually, for small triangles, they are similar.
            
            # Let's use the side lengths.
            # If d12^2 > d23^2 + d31^2 (approx), it's obtuse at p3?
            # Spherical Law of Cosines: cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
            # If A > 90, cos(A) < 0, so cos(a) < cos(b)cos(c).
            
            # Let's check all three angles.
            def is_acute(p1, p2, p3, d12, d23, d31):
                # Check angle at p1
                cos_a1 = np.clip(np.dot(p2, p3), -1.0, 1.0) # This is cos(d23)
                # Law of Cosines for sides: cos(d23) = cos(d12)cos(d31) + sin(d12)sin(d31)cos(A1)
                # cos(A1) = (cos(d23) - cos(d12)cos(d31)) / (sin(d12)sin(d31))
                sin_d12 = np.sin(d12)
                sin_d31 = np.sin(d31)
                if sin_d12 < 1e-10 or sin_d31 < 1e-10:
                    return False
                cos_A1 = (np.cos(d23) - np.cos(d12)*np.cos(d31)) / (sin_d12 * sin_d31)
                cos_A1 = np.clip(cos_A1, -1.0, 1.0)
                if cos_A1 < 0: return False # Angle > 90
                
                # Check angle at p2
                sin_d23 = np.sin(d23)
                if sin_d23 < 1e-10 or sin_d31 < 1e-10:
                    return False
                cos_A2 = (np.cos(d31) - np.cos(d12)*np.cos(d23)) / (sin_d12 * sin_d23)
                cos_A2 = np.clip(cos_A2, -1.0, 1.0)
                if cos_A2 < 0: return False
                
                # Check angle at p3
                sin_d12 = np.sin(d12)
                sin_d23 = np.sin(d23)
                if sin_d12 < 1e-10 or sin_d23 < 1e-10:
                    return False
                cos_A3 = (np.cos(d12) - np.cos(d23)*np.cos(d31)) / (sin_d23 * sin_d31)
                cos_A3 = np.clip(cos_A3, -1.0, 1.0)
                if cos_A3 < 0: return False
                
                return True
            
            if is_acute(p1, p2, p3, d12, d23, d31):
                current_radius = rc
            else:
                max_side = max(d12, d23, d31)
                current_radius = max_side / 2.0
                
        if current_radius > max_radius:
            max_radius = current_radius
            
    return max_radius

def _local_search(points, max_iter=1000, step_size=0.1):
    """
    Local search to minimize the covering radius.
    """
    points = _normalize(points).copy()
    current_radius = _covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    for i in range(max_iter):
        # Pick a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Move it slightly
        # Use a small random perturbation
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = _normalize(new_point)
        
        points[idx] = new_point
        new_radius = _covering_radius(points)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            points[idx] = original_point
            
        # Decrease step size
        step_size *= 0.999
        
    return best_points

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Local Search
    points_final = _local_search(x0, max_iter=5000, step_size=0.5)
    
    # 3. Final check and normalization
    points_final = _normalize(points_final)
    
    return points_final
# EVOLVE-BLOCK-END
