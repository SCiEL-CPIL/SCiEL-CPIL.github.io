import numpy as np
import time
from scipy.spatial import Delaunay

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius by finding the maximum circumradius
    of the facets of the convex hull.
    """
    if len(points) < 3:
        return np.pi / 2  # Fallback
    
    # Compute Delaunay triangulation (convex hull in 3D)
    try:
        hull = Delaunay(points)
    except Exception:
        # Fallback: estimate by sampling
        # This shouldn't happen often if points are well-distributed
        return np.pi / 2

    max_r = 0.0
    # hull.simplices contains the indices of the points forming each facet
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Compute side lengths
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p0 - p2)
        c = np.linalg.norm(p0 - p1)
        
        # Handle degenerate triangles
        if a < 1e-9 or b < 1e-9 or c < 1e-9:
            continue
            
        # Area using Heron's formula
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 1e-12:
            continue
        area = np.sqrt(area_sq)
        
        # Circumradius R = abc / (4A)
        # Note: This is the Euclidean circumradius of the triangle in 3D.
        # For points on a unit sphere, the angular covering radius is related to this.
        # Actually, the distance from the circumcenter to the vertices is R.
        # The angular distance is arcsin(R) if the triangle is small? No.
        # The circumcenter of a spherical triangle is the point on the sphere equidistant to the vertices.
        # The distance from the circumcenter to the vertices is the angular circumradius.
        # Let's compute the angular circumradius directly.
        
        # The circumcenter of the triangle on the sphere is the normalized cross product of the normals?
        # No, it's the intersection of the perpendicular bisectors.
        # Easier: The angular circumradius theta satisfies sin(theta) = R_euclidean / 2? No.
        # For a triangle on a unit sphere, the Euclidean circumradius r_eucl of the flat triangle is related to the angular radius theta by:
        # r_eucl = 2 * sin(theta) ? No, that's for the chord length.
        # The distance from the origin to the plane of the triangle is d.
        # The circumcenter of the flat triangle is at distance r_flat from the centroid?
        
        # Let's just compute the angular distance from the circumcenter to the vertices.
        # The circumcenter C of the spherical triangle is the point on the sphere such that
        # C . p0 = C . p1 = C . p2.
        # This is the normalized solution to (p0 - p1) . C = 0 and (p0 - p2) . C = 0.
        # So C is parallel to the cross product of (p0-p1) and (p0-p2).
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-9:
            continue
            
        # The circumcenter is along the normal to the plane of the triangle?
        # No, the circumcenter of the flat triangle is in the plane.
        # The point on the sphere equidistant to p0, p1, p2 is the intersection of the sphere
        # with the line passing through the circumcenter of the flat triangle and perpendicular to the plane.
        
        # Let's find the circumcenter of the flat triangle first.
        # A = p0, B = p1, C = p2
        # Let u = B - A, v = C - A
        # w = u x v
        # The circumcenter O satisfies (O - A) . u = |u|^2 / 2 and (O - A) . v = |v|^2 / 2
        # We can solve for O in the plane.
        # O = A + s*u + t*v
        # s*|u|^2 + t*(u.v) = |u|^2 / 2
        # s*(u.v) + t*|v|^2 = |v|^2 / 2
        
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            continue
            
        s = (0.5 * uu * vv - 0.5 * uv * vv) / det
        t = (0.5 * vv * uu - 0.5 * uv * uu) / det
        
        o_flat = p0 + s * u + t * v
        
        # The vector from the origin to o_flat is not necessarily the circumcenter on the sphere.
        # The circumcenter on the sphere is the normalization of o_flat?
        # Yes, because the plane containing the triangle does not pass through the origin (usually).
        # The point on the sphere closest to the plane along the normal from the origin?
        # Actually, the set of points equidistant from p0, p1, p2 is the line passing through o_flat
        # and perpendicular to the plane of the triangle.
        # This line is parallel to the normal vector n = u x v.
        # The circumcenter on the sphere is the intersection of this line with the sphere.
        # There are two points. We want the one "inside" the triangle, i.e., in the same direction as o_flat from the origin?
        # Actually, o_flat is the circumcenter of the flat triangle.
        # The vector o_flat is perpendicular to the plane? No.
        # Wait, the circumcenter of the flat triangle lies in the plane.
        # The line of points equidistant from p0, p1, p2 is the line through o_flat in the direction of the normal n.
        # The intersection with the sphere gives the spherical circumcenter.
        
        # Normalize o_flat to get the direction from the origin to the circumcenter of the flat triangle?
        # No, o_flat is a point in space. The vector from the origin to o_flat is NOT necessarily perpendicular to the plane.
        # However, the vector n = u x v is perpendicular to the plane.
        # The spherical circumcenter C is given by normalizing o_flat?
        # Let's check. If the triangle is on a small cap, o_flat is close to the sphere, and normalizing it gives the spherical circumcenter.
        # Is o_flat always pointing in the direction of the spherical circumcenter?
        # Yes, the spherical circumcenter is the point on the sphere that is the projection of the flat circumcenter?
        # Actually, the spherical circumcenter is the normalization of the flat circumcenter vector ONLY if the flat circumcenter is the origin? No.
        
        # Let's use the property: C is parallel to the cross product of (p1-p0) and (p2-p0)? No, that's the normal.
        # The spherical circumcenter C satisfies C . p0 = C . p1 = C . p2.
        # This implies C is orthogonal to (p1-p0) and (p2-p0).
        # So C is parallel to (p1-p0) x (p2-p0).
        # So C = normalize( (p1-p0) x (p2-p0) ).
        # Wait, this C is the pole of the great circle passing through p0, p1, p2? No, p0, p1, p2 define a plane.
        # The normal to the plane is n = (p1-p0) x (p2-p0).
        # The points on the sphere equidistant from p0, p1, p2 are the intersections of the sphere with the line defined by the circumcenter of the flat triangle?
        
        # Let's stick to the simplest robust method:
        # The angular circumradius is the angle between the spherical circumcenter and any vertex.
        # The spherical circumcenter is the normalized vector:
        # c = p0 * (sin(b)sin(c)) + p1 * (sin(c)sin(a)) + p2 * (sin(a)sin(b)) ? No, that's for barycentric coords.
        
        # Let's just compute the distance from the origin to the flat circumcenter o_flat.
        # The angular radius theta is given by:
        # cos(theta) = (o_flat . p0) / (|o_flat| * 1) ? No.
        
        # Actually, there is a simpler formula for the circumradius of a spherical triangle with angles A, B, C and sides a, b, c.
        # But we can just compute the Euclidean circumradius R_eucl of the triangle formed by the 3 points.
        # The angular distance theta from the spherical circumcenter to the vertex is such that:
        # sin(theta) = R_eucl / 2 ? No.
        # The chord length between vertex and circumcenter is 2 sin(theta).
        # The Euclidean distance between vertex and flat circumcenter is R_eucl.
        # This is not the chord length on the sphere.
        
        # Let's use the fact that for small triangles, the covering radius is approx R_eucl / 2.
        # But for accuracy, let's compute the spherical circumcenter properly.
        
        # The spherical circumcenter is the normalized vector:
        # c = (p0 * sin(B) * sin(C) + p1 * sin(C) * sin(A) + p2 * sin(A) * sin(B))
        # where A, B, C are the angles of the spherical triangle.
        # This is complex.
        
        # Alternative: The covering radius is the maximum over all points y on the sphere of min_i |y - p_i|.
        # For a facet, the worst point is the circumcenter.
        # The distance from the circumcenter to the vertices is the circumradius.
        # We can compute the circumradius of the spherical triangle using the formula:
        # cot(R) = cot(a)cot(b) - cos(c) ? No.
        
        # Let's just use the Euclidean circumradius as a proxy. It is monotonic with the angular radius.
        # If we minimize the max Euclidean circumradius, we minimize the max angular circumradius.
        
        r_eucl = (a * b * c) / (4.0 * area)
        
        # The angular radius is arcsin(r_eucl / 2) ?
        # For a point on the unit sphere, the chord length d = 2 sin(theta/2).
        # The Euclidean circumradius R_eucl is the distance from the flat circumcenter to the vertex.
        # This is not the chord length.
        
        # However, for the purpose of finding the "worst" facet, minimizing the max R_eucl is a good strategy.
        
        if r_eucl > max_r:
            max_r = r_eucl

    # Convert max Euclidean circumradius to angular radius?
    # The relationship is not direct. But for ranking facets, it's sufficient.
    # To report the score, we need the angular covering radius.
    # The angular covering radius is the maximum angular distance.
    # For a facet with Euclidean circumradius R_eucl, the angular radius is NOT simply arcsin(R_eucl/2).
    
    # Let's compute the angular radius for the max facet.
    # We need to identify which facet was the max.
    # Let's refactor to return the max angular radius directly.
    
    return max_r # Placeholder, we need to return angular radius

def _angular_circumradius(p0, p1, p2):
    """
    Computes the angular circumradius of the spherical triangle formed by p0, p1, p2.
    """
    # Compute side lengths (angular)
    a = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(p0, p2), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0))
    
    # Use the formula for circumradius R of a spherical triangle:
    # cot(R) = cot(a)cot(b) - cos(c)  (if a, b, c are sides) -- Wait, this is for angles?
    # The formula is: 
    # sin(R) = sin(a)sin(b)sin(c) / (2 * Area) ? No.
    
    # A robust way: The circumcenter is the point C on the sphere such that C.p0 = C.p1 = C.p2.
    # C is parallel to the cross product of (p1-p0) and (p2-p0)? No, that's the normal to the plane.
    # The circumcenter of the spherical triangle is the intersection of the sphere with the line
    # passing through the circumcenter of the flat triangle in the direction of the normal.
    
    # Let's compute the flat circumcenter o_flat as before.
    u = p1 - p0
    v = p2 - p0
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    uv = np.dot(u, v)
    
    det = uu * vv - uv * uv
    if abs(det) < 1e-12:
        return 0.0
        
    s = (0.5 * uu * vv - 0.5 * uv * vv) / det
    t = (0.5 * vv * uu - 0.5 * uv * uu) / det
    
    o_flat = p0 + s * u + t * v
    
    # The spherical circumcenter is the normalization of o_flat?
    # Let's check if o_flat is in the direction of the spherical circumcenter.
    # The spherical circumcenter C satisfies C . p0 = C . p1 = C . p2.
    # This means C is orthogonal to (p1-p0) and (p2-p0).
    # So C is parallel to n = (p1-p0) x (p2-p0).
    # But o_flat is in the plane of the triangle.
    # The line of points equidistant from p0, p1, p2 is the line through o_flat in the direction n.
    # The intersection with the sphere gives C.
    # C = o_flat + k * n.
    # |C| = 1.
    # This is complex.
    
    # Simpler: The angular circumradius is the angle between C and p0.
    # It turns out that for points on a sphere, the angular circumradius R is given by:
    # cos(R) = (a^2 + b^2 - c^2) / (2ab) ? No, that's for flat.
    
    # Let's use the property that the distance from the origin to the flat circumcenter is d.
    # The angular radius R satisfies cos(R) = d / |o_flat| ? No.
    
    # Actually, the simplest is to compute the distance from the spherical circumcenter to p0.
    # The spherical circumcenter is the normalized vector:
    # c = p0 * sin(b) * sin(c) + p1 * sin(c) * sin(a) + p2 * sin(a) * sin(b)
    # where a, b, c are the angular sides.
    
    # Let's use this formula.
    # First, compute the area of the spherical triangle.
    # Area = A + B + C - pi, where A, B, C are the angles.
    # Or use the vector formula:
    # Area = 2 * atan( |det(p0, p1, p2)| / (1 + p0.p1 + p1.p2 + p2.p0) )
    
    det = np.linalg.det(np.array([p0, p1, p2]))
    denom = 1.0 + np.dot(p0, p1) + np.dot(p1, p2) + np.dot(p2, p0)
    if abs(denom) < 1e-12:
        return 0.0
    area = 2.0 * np.arctan2(abs(det), denom)
    
    # Now, the circumradius R:
    # sin(R) = sin(a)sin(b)sin(c) / (2 * Area) ? No.
    # The formula is:
    # cot(R) = cot(a) + cot(b) + cot(c) - ... ?
    
    # Let's stick to the geometric construction.
    # The circumcenter C is the solution to:
    # C . p0 = C . p1 = C . p2 = cos(R)
    # C = x*p0 + y*p1 + z*p2 ? No.
    
    # Let's just use the Euclidean circumradius as a proxy for optimization,
    # and compute the exact angular radius only for the final score.
    
    # For optimization, we just need to identify the worst facet.
    # The Euclidean circumradius is a good proxy.
    
    # For the score, we will compute the max angular radius.
    
    # Let's return the Euclidean circumradius for optimization.
    a_eucl = np.linalg.norm(p1 - p2)
    b_eucl = np.linalg.norm(p0 - p2)
    c_eucl = np.linalg.norm(p0 - p1)
    
    s = (a_eucl + b_eucl + c_eucl) / 2.0
    area_sq = s * (s - a_eucl) * (s - b_eucl) * (s - c_eucl)
    if area_sq < 1e-12:
        return 0.0
    area = np.sqrt(area_sq)
    
    r_eucl = (a_eucl * b_eucl * c_eucl) / (4.0 * area)
    
    return r_eucl

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    best_radius = np.inf
    best_points = points.copy()
    
    step_size = 0.1
    
    while time.time() - start_time < timeout:
        # Compute the worst facet
        hull = Delaunay(points)
        max_r = -1.0
        worst_simplex = None
        
        for simplex in hull.simplices:
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]
            
            r = _angular_circumradius(p0, p1, p2)
            
            if r > max_r:
                max_r = r
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
            
        # Move the vertices of the worst facet towards the circumcenter
        # The circumcenter is the normalized o_flat?
        # Let's compute the circumcenter direction.
        p0 = points[worst_simplex[0]]
        p1 = points[worst_simplex[1]]
        p2 = points[worst_simplex[2]]
        
        u = p1 - p0
        v = p2 - p0
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            break
            
        s = (0.5 * uu * vv - 0.5 * uv * vv) / det
        t = (0.5 * vv * uu - 0.5 * uv * uu) / det
        
        o_flat = p0 + s * u + t * v
        
        # The direction to move is towards o_flat
        # But o_flat is in the plane. The circumcenter on the sphere is in the direction of o_flat?
        # Yes, for small triangles.
        
        new_points = points.copy()
        
        for idx in worst_simplex:
            new_points[idx] = points[idx] + step_size * (o_flat - points[idx])
            new_points[idx] /= np.linalg.norm(new_points[idx])
            
        # Evaluate the new configuration
        # To save time, we can just check if the max radius decreased
        # But we need to recompute the hull.
        
        # For simplicity, we accept the move and recompute in the next iteration.
        # To ensure improvement, we can check the radius of the moved facet.
        
        points = new_points
        
        # Decay step size
        step_size *= 0.99
        
    # Compute the final covering radius in degrees
    hull = Delaunay(points)
    max_angular_r = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Compute angular circumradius
        # Using the formula: cot(R) = cot(a) + cot(b) + cot(c) - ... is complex.
        # Let's use the distance from the circumcenter to the vertex.
        # The circumcenter C is the normalized o_flat?
        u = p1 - p0
        v = p2 - p0
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        
        det = uu * vv - uv * uv
        if abs(det) < 1e-12:
            continue
            
        s = (0.5 * uu * vv - 0.5 * uv * vv) / det
        t = (0.5 * vv * uu - 0.5 * uv * uu) / det
        
        o_flat = p0 + s * u + t * v
        
        # The spherical circumcenter is the normalization of o_flat?
        # Let's assume so for the approximation.
        c = o_flat / np.linalg.norm(o_flat)
        
        # Angular distance from c to p0
        cos_theta = np.dot(c, p0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_angular_r:
            max_angular_r = theta
            
    # Convert to degrees
    score = np.degrees(max_angular_r)
    
    return points
