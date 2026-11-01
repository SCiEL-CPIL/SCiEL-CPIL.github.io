"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _angular_distance(p, q):
    # p, q are unit vectors
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)

def _circumcenter_sphere(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle.
    Returns the unit vector of the circumcenter and the radius (angular distance to vertices).
    """
    # The circumcenter is equidistant from p1, p2, p3.
    # It lies in the plane defined by the cross products of the vertices.
    # Specifically, c = (p1 x p2) x (p2 x p3) + ... is not quite right.
    # A robust way: The circumcenter c satisfies c . p1 = c . p2 = c . p3 = cos(R).
    # This implies c is orthogonal to (p1 - p2) and (p2 - p3) in the tangent space? No.
    # c is in the span of p1, p2, p3.
    # Let c = a*p1 + b*p2 + c*p3.
    # c . p1 = c . p2 => a + b(p1.p2) + c(p3.p1) = b(p1.p2) + 1 + c(p2.p3) ... this is messy.
    
    # Simpler: The circumcenter is the normalized vector:
    # c = (p1 x p2) x (p2 x p3) + (p2 x p3) x (p3 x p1) + (p3 x p1) x (p1 x p2)
    # This vector is orthogonal to the plane of the triangle? No, it's the pole of the great circle?
    
    # Let's use the property that the circumcenter is the intersection of the perpendicular bisectors.
    # Bisector of p1, p2: points x such that x.p1 = x.p2.
    # This is the great circle through the midpoint of the arc p1-p2 and the pole of the plane p1-p2?
    
    # Alternative: Use the formula for the circumcenter of a spherical triangle.
    # c = ( (sin(A) * p1) + (sin(B) * p2) + (sin(C) * p3) ) / norm
    # where A, B, C are the angles of the triangle at vertices p1, p2, p3.
    
    # Let's compute the angles.
    # Side lengths (angular)
    a = _angular_distance(p2, p3)
    b = _angular_distance(p1, p3)
    c_ = _angular_distance(p1, p2)
    
    # Spherical law of cosines for angles:
    # cos(A) = (cos(a) - cos(b)cos(c_)) / (sin(b)sin(c_))
    cos_a = np.cos(a)
    cos_b = np.cos(b)
    cos_c = np.cos(c_)
    sin_b = np.sin(b)
    sin_c = np.sin(c_)
    
    # Handle degenerate cases
    if sin_b * sin_c < 1e-10:
        # Degenerate triangle, return midpoint or one of the vertices
        return _normalize(p1 + p2 + p3), 0.0
        
    cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
    cos_A = np.clip(cos_A, -1.0, 1.0)
    A = np.arccos(cos_A)
    
    # Similarly for B and C
    cos_B = (cos_b - cos_a * cos_c) / (np.sin(a) * sin_c)
    cos_B = np.clip(cos_B, -1.0, 1.0)
    B = np.arccos(cos_B)
    
    cos_C = (cos_c - cos_a * cos_b) / (np.sin(a) * sin_b)
    cos_C = np.clip(cos_C, -1.0, 1.0)
    C = np.arccos(cos_C)
    
    # Circumcenter vector
    c_vec = np.sin(A) * p1 + np.sin(B) * p2 + np.sin(C) * p3
    c_norm = np.linalg.norm(c_vec)
    if c_norm < 1e-10:
        # Degenerate, e.g., antipodal points
        return _normalize(p1 + p2 + p3), 0.0
    c_unit = c_vec / c_norm
    
    # Radius is the angular distance from circumcenter to any vertex
    radius = _angular_distance(c_unit, p1)
    
    return c_unit, radius

def _is_point_in_spherical_triangle(p, p1, p2, p3):
    """
    Check if point p is inside the spherical triangle p1, p2, p3.
    We check if p is on the same side of each edge as the opposite vertex.
    """
    # Edge p1-p2, opposite p3
    # Normal to the great circle through p1, p2 is n12 = p1 x p2
    # p is on the same side as p3 if (p x n12) . p3 > 0 ?
    # Let's use the dot product with the normal of the plane containing the edge and the center?
    # A robust method: barycentric coordinates on the sphere.
    # Or simply: The triangle is the intersection of three hemispheres.
    # Hemisphere 1: defined by edge p1-p2, containing p3.
    # Normal n12 = normalize(cross(p1, p2))
    # The hemisphere is { x | dot(x, n12) >= 0 } if dot(p3, n12) > 0, else <= 0.
    
    def _side(p_edge1, p_edge2, p_test, p_ref):
        n = np.cross(p_edge1, p_edge2)
        n = _normalize(n)
        s_test = np.dot(p_test, n)
        s_ref = np.dot(p_ref, n)
        return (s_test * s_ref) >= 0

    if not _side(p1, p2, p, p3): return False
    if not _side(p2, p3, p, p1): return False
    if not _side(p3, p1, p, p2): return False
    return True

def _covering_radius(points):
    """
    Compute the covering radius and the worst facet (indices).
    """
    # Convex hull of points on sphere gives the facets
    # scipy.spatial.ConvexHull works in 3D
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fallback to max distance to nearest neighbor
        # This is a rough estimate
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points)
        return np.pi, 0, 1, 2 # Max possible radius, dummy indices

    max_radius = 0.0
    worst_i, worst_j, worst_k = 0, 1, 2
    worst_cc = None
    
    for simplex in hull.simplices:
        i, j, k = simplex
        p1, p2, p3 = points[i], points[j], points[k]
        cc, r = _circumcenter_sphere(p1, p2, p3)
        
        # Check if circumcenter is inside the triangle
        # If not, the worst point in this facet is one of the vertices
        # The radius of the empty cap is the max distance to the vertices
        if _is_point_in_spherical_triangle(cc, p1, p2, p3):
            current_r = r
        else:
            # If circumcenter is outside, the "worst" point is the vertex farthest from the others?
            # Actually, the covering radius for the facet is the max of the distances from the vertices to the other vertices?
            # No, the empty cap is bounded by the arcs. The center of the largest empty circle in a triangle
            # is either the circumcenter (if acute) or the midpoint of the longest side (if obtuse).
            # For an obtuse spherical triangle, the circumcenter is outside.
            # The largest empty circle center is the midpoint of the longest side.
            d12 = _angular_distance(p1, p2)
            d23 = _angular_distance(p2, p3)
            d31 = _angular_distance(p3, p1)
            
            if d12 >= d23 and d12 >= d31:
                current_r = d12 / 2.0
            elif d23 >= d12 and d23 >= d31:
                current_r = d23 / 2.0
            else:
                current_r = d31 / 2.0
                
        if current_r > max_radius:
            max_radius = current_r
            worst_i, worst_j, worst_k = i, j, k
            worst_cc = cc if _is_point_in_spherical_triangle(cc, p1, p2, p3) else None
            
    return max_radius, worst_i, worst_j, worst_k

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Local Search Optimization
    # We will iterate, finding the worst facet and pushing the vertices away from the circumcenter.
    
    start_time = time.time()
    time_limit = 110.0 # Leave some time for evaluation
    
    step_size = 0.01
    patience = 0
    max_patience = 50
    prev_radius = _covering_radius(points)[0]
    
    while time.time() - start_time < time_limit:
        radius, i, j, k = _covering_radius(points)
        
        if radius < prev_radius:
            prev_radius = radius
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        p1, p2, p3 = points[i], points[j], points[k]
        
        # Compute circumcenter of the worst facet
        cc, r_cc = _circumcenter_sphere(p1, p2, p3)
        
        # Check if cc is inside the triangle
        if not _is_point_in_spherical_triangle(cc, p1, p2, p3):
            # If outside, we should move the two vertices of the longest side apart
            # For simplicity, we just push all three away from the centroid if it's bad?
            # Or better: identify the longest side and push its endpoints.
            d12 = _angular_distance(p1, p2)
            d23 = _angular_distance(p2, p3)
            d31 = _angular_distance(p3, p1)
            
            # Find longest side
            if d12 >= d23 and d12 >= d31:
                # Push p1 and p2 apart
                # Direction for p1: away from p2
                # Tangent vector at p1 away from p2
                t1 = _normalize(p2 - (np.dot(p1, p2)*p1))
                t1 = -t1 # Away from p2
                points[i] = _normalize(p1 + step_size * t1)
                
                t2 = _normalize(p1 - (np.dot(p2, p1)*p2))
                t2 = -t2 # Away from p1
                points[j] = _normalize(p2 + step_size * t2)
            elif d23 >= d12 and d23 >= d31:
                # Push p2 and p3 apart
                t2 = _normalize(p3 - (np.dot(p2, p3)*p2))
                t2 = -t2
                points[j] = _normalize(p2 + step_size * t2)
                
                t3 = _normalize(p2 - (np.dot(p3, p2)*p3))
                t3 = -t3
                points[k] = _normalize(p3 + step_size * t3)
            else:
                # Push p3 and p1 apart
                t3 = _normalize(p1 - (np.dot(p3, p1)*p3))
                t3 = -t3
                points[k] = _normalize(p3 + step_size * t3)
                
                t1 = _normalize(p3 - (np.dot(p1, p3)*p1))
                t1 = -t1
                points[i] = _normalize(p1 + step_size * t1)
        else:
            # Circumcenter is inside. Push all three vertices away from the circumcenter.
            # This expands the triangle, reducing the circumradius.
            
            # Tangent vector at p1 away from cc
            # Project cc onto tangent space of p1?
            # t1 = cc - (cc.p1)p1, then normalize.
            proj1 = cc - np.dot(cc, p1) * p1
            t1 = _normalize(proj1)
            points[i] = _normalize(p1 + step_size * t1)
            
            proj2 = cc - np.dot(cc, p2) * p2
            t2 = _normalize(proj2)
            points[j] = _normalize(p2 + step_size * t2)
            
            proj3 = cc - np.dot(cc, p3) * p3
            t3 = _normalize(proj3)
            points[k] = _normalize(p3 + step_size * t3)
            
        # Reduce step size slowly if stuck?
        # For now, keep it simple.
        
    return points
# EVOLVE-BLOCK-END
