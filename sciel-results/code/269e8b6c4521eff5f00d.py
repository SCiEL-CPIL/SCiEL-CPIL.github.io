"""Rewritten program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

N = 130

def _angular_distance_deg(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    Uses the convex hull method: the covering radius is the maximum circumradius
    of the facets of the convex hull of the points."""
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Check for coincident points
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(points, points)
    if np.min(dists) < 1e-10:
        return 90.0  # Invalid: coincident points
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v such that v·p >= 0 for all p
    # This is equivalent to the origin being outside the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    # Check if origin is inside the convex hull
    # If the origin is inside, the points are NOT in a closed hemisphere
    # We can check by seeing if all facet normals point away from the origin
    # For a facet with vertices v0, v1, v2, the normal n should satisfy n·v0 > 0 (for outward normal)
    # If the origin is inside, then for every facet, the origin is on the same side as the interior
    
    # Compute circumradius for each facet
    max_circumradius = 0.0
    for simplex in hull.simplices:
        facet = points[simplex]
        # For a triangle, the circumradius R = abc / (4 * Area)
        # where a, b, c are side lengths
        a = np.linalg.norm(facet[1] - facet[2])
        b = np.linalg.norm(facet[0] - facet[2])
        c = np.linalg.norm(facet[0] - facet[1])
        
        # Area of triangle
        cross = np.cross(facet[1] - facet[0], facet[2] - facet[0])
        area = 0.5 * np.linalg.norm(cross)
        
        if area < 1e-12:
            # Degenerate facet
            # Use the maximum distance from the center of the facet to its vertices
            centroid = np.mean(facet, axis=0)
            max_dist = np.max(np.linalg.norm(facet - centroid, axis=1))
            # Angular distance
            cos_theta = np.clip(1.0 - max_dist**2 / 2.0, -1.0, 1.0)
            theta = math.degrees(math.acos(cos_theta))
            max_circumradius = max(max_circumradius, theta)
            continue
        
        circumradius = (a * b * c) / (4.0 * area)
        # The circumradius in angular terms: for points on the unit sphere,
        # the circumradius of the spherical triangle is related to the planar circumradius
        # Actually, the covering radius for a facet is the angular distance from the facet center to the farthest vertex
        # For a spherical triangle with side lengths a, b, c (angular), the circumradius is...
        # Let me use a different approach: the circumcenter of the planar triangle projected to the sphere
        
        # Simpler: the maximum angular distance from any point in the facet to the nearest configuration point
        # is the circumradius of the spherical triangle
        # For a spherical triangle, the circumradius R satisfies:
        # cos(R) = (cos(a)cos(b)cos(c) + 2cos(a)cos(b)cos(c)) / (1 - cos^2(a) - cos^2(b) - cos^2(c) + 2cos(a)cos(b)cos(c))
        # This is complex. Let me use the planar approximation or compute directly.
        
        # Actually, the correct formula: for a spherical triangle with angular sides a, b, c,
        # the circumradius R is given by:
        # sin(R) = sin(a)sin(b)sin(c) / (2 * Area_spherical)
        # where Area_spherical = A + B + C - pi (spherical excess)
        
        # Let me compute the spherical excess
        # Using the formula: tan(A/2) = sqrt(tan(s/2)tan((s-a)/2)tan((s-b)/2)tan((s-c)/2))
        # where s = (a+b+c)/2
        
        a_ang = math.acos(np.clip(np.dot(facet[1], facet[2]), -1.0, 1.0))
        b_ang = math.acos(np.clip(np.dot(facet[0], facet[2]), -1.0, 1.0))
        c_ang = math.acos(np.clip(np.dot(facet[0], facet[1]), -1.0, 1.0))
        
        s = (a_ang + b_ang + c_ang) / 2.0
        if s < 1e-10 or s > math.pi - 1e-10:
            max_circumradius = max(max_circumradius, 90.0)
            continue
        
        # Spherical excess
        # Using L'Huilier's formula: tan(E/4) = sqrt(tan(s/2)tan((s-a)/2)tan((s-b)/2)tan((s-c)/2))
        tan_s2 = math.tan(s / 2.0)
        tan_sa2 = math.tan((s - a_ang) / 2.0)
        tan_sb2 = math.tan((s - b_ang) / 2.0)
        tan_sc2 = math.tan((s - c_ang) / 2.0)
        
        if tan_s2 * tan_sa2 * tan_sb2 * tan_sc2 < 0:
            max_circumradius = max(max_circumradius, 90.0)
            continue
        
        E = 4.0 * math.atan(math.sqrt(max(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2, 0.0)))
        
        # Circumradius
        # sin(R) = sin(a)sin(b)sin(c) / (2 * sin(E/2) * ... )
        # Actually, for a spherical triangle, the circumradius R satisfies:
        # sin(R) = sin(a)sin(b)sin(c) / (4 * sin(E/2) * sin((A+B+C)/2))
        # This is getting complicated. Let me use a numerical approach.
        
        # Numerical: find the point on the sphere that is equidistant from all three vertices
        # and maximizes the minimum distance to the three vertices
        # This is the circumcenter of the spherical triangle
        
        # For now, use the planar circumradius as an approximation
        # The angular circumradius is approximately arcsin(circumradius_planar) for small triangles
        # But this is not accurate. Let me use a direct computation.
        
        # The circumcenter of a spherical triangle can be found by intersecting the perpendicular bisectors
        # In 3D, the circumcenter is the intersection of the plane containing the three points 
        # with the line through the origin perpendicular to that plane (for the planar case)
        # For the spherical case, it's the normalization of the planar circumcenter
        
        # Planar circumcenter of the triangle formed by the three points
        # For points p1, p2, p3 in 3D, the circumcenter is:
        # C = p1 + (|p1-p2|^2 * (p1-p3) x (n) + |p1-p3|^2 * (n) x (p1-p2)) / (2 * |n|^2 * ... )
        # This is complex. Let me use a simpler approach.
        
        # The normal to the plane of the facet
        n = np.cross(facet[1] - facet[0], facet[2] - facet[0])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            max_circumradius = max(max_circumradius, 90.0)
            continue
        n = n / n_norm
        
        # The circumcenter in the plane is at:
        # C = centroid + t * n where t is chosen so that |C - p1| = |C - p2| = |C - p3|
        # |C - p1|^2 = |C - p2|^2
        # |C + t*n - p1|^2 = |C + t*n - p2|^2
        # |C-p1|^2 + 2t*n·(C-p1) + t^2 = |C-p2|^2 + 2t*n·(C-p2) + t^2
        # |C-p1|^2 - |C-p2|^2 + 2t*n·(C-p1 - C+p2) = 0
        # |C-p1|^2 - |C-p2|^2 + 2t*n·(p2-p1) = 0
        
        centroid = np.mean(facet, axis=0)
        d12 = np.dot(centroid - facet[0], centroid - facet[0]) - np.dot(centroid - facet[1], centroid - facet[1])
        d13 = np.dot(centroid - facet[0], centroid - facet[0]) - np.dot(centroid - facet[2], centroid - facet[2])
        
        # Actually, let me just use the formula for the circumcenter of a triangle in 3D
        # Project to 2D or use the vector formula
        
        # Simpler: the circumcenter C satisfies:
        # |C - p_i|^2 = R^2 for i=0,1,2
        # C·C - 2C·p_i + p_i·p_i = R^2
        # Since |p_i| = 1: C·C - 2C·p_i + 1 = R^2
        # So C·p_i = (C·C + 1 - R^2)/2 = constant for all i
        # This means C is in the direction of the vector that is equidistant from all p_i in dot product
        
        # The circumcenter is along the normal n, so C = alpha * n
        # alpha * n · p_i = constant
        # alpha * (n · p_i) = constant
        # n · p_i should be the same for all i if the triangle is isosceles... no.
        
        # Let me just compute it numerically
        # The circumcenter of the spherical triangle is the point on the sphere that maximizes the minimum distance to the three vertices
        # This is equivalent to finding the center of the smallest circle on the sphere containing the three points
        
        # For a triangle on the sphere, the circumradius can be computed as:
        # R = arccos( (cos(a)cos(b) + cos(c)) / (1 + cos(a)cos(b)cos(c) - cos^2(c)) )
        # No, that's not right either.
        
        # Let me use the formula: for a spherical triangle with angular sides a, b, c,
        # the circumradius R satisfies:
        # cos(R) = (cos(a)cos(b)cos(c) + 2cos(a)cos(b)cos(c)) / sqrt(1 - 2cos^2(a)cos^2(b) - 2cos^2(a)cos^2(c) - 2cos^2(b)cos^2(c) + 4cos^2(a)cos^2(b)cos^2(c))
        
        # This is getting too complex. Let me use a numerical optimization to find the circumradius.
        
        # Actually, let me just use the fact that the covering radius is the maximum over all facets
        # of the maximum angular distance from the facet to the nearest point.
        # For a spherical triangle, this is the circumradius.
        
        # I'll use a simple numerical approach: sample points on the spherical triangle
        # and find the maximum minimum distance to the three vertices.
        
        # For efficiency, let me just use the planar circumradius as an upper bound
        # and refine if needed.
        
        # Planar circumradius
        if area > 1e-12:
            circumradius_planar = (a * b * c) / (4.0 * area)
            # The angular circumradius is arcsin(min(circumradius_planar, 1.0)) approximately
            # But this is not accurate for large triangles.
            
            # Let me compute the actual spherical circumradius using the formula:
            # For a spherical triangle with sides a, b, c (angular), the circumradius R is:
            # sin(R) = sin(a)sin(b)sin(c) / (2 * sin(A) * sin(B) * sin(C) / sin(E))
            # where A, B, C are the angles of the triangle and E is the spherical excess.
            
            # This is still complex. Let me just use a numerical search.
            
            # Find the point on the sphere that is equidistant from all three vertices
            # This is the circumcenter.
            
            # The circumcenter of a spherical triangle lies at the intersection of the perpendicular bisectors.
            # In 3D, it's the point C on the unit sphere such that C·p1 = C·p2 = C·p3.
            # This means C is orthogonal to (p1-p2) and (p1-p3).
            # So C is parallel to (p1-p2) x (p1-p3) = n (the normal to the plane).
            # So C = ±n / |n|
            
            # But wait, this is only true if the triangle is right-angled or something.
            # Actually, C·p1 = C·p2 implies C·(p1-p2) = 0, so C is orthogonal to p1-p2.
            # Similarly C is orthogonal to p1-p3.
            # So C is parallel to (p1-p2) x (p1-p3) = n.
            # So C = n / |n| or C = -n / |n|.
            
            # Let me check which one gives the correct circumradius.
            # The circumradius R satisfies cos(R) = C·p1 = C·p2 = C·p3.
            
            C1 = n / np.linalg.norm(n)
            C2 = -C1
            
            cos_R1 = np.dot(C1, facet[0])
            cos_R2 = np.dot(C2, facet[0])
            
            # The circumcenter should be on the same side of the plane as the interior of the sphere
            # For a convex hull facet, the normal n points outward, so the interior is in the direction -n
            # So the circumcenter should be C2 = -n / |n| if the origin is inside the convex hull
            
            # Check if the origin is inside the convex hull
            # For each facet, the origin should be on the same side as the interior
            # The interior is in the direction of -n (since n points outward)
            # So the origin should satisfy n·0 < 0, which is always true.
            
            # The circumcenter is the point that is equidistant from all three vertices
            # and lies on the sphere. Both C1 and C2 are on the sphere.
            # We need to check which one is the circumcenter of the spherical triangle.
            
            # The circumradius R is the angular distance from the circumcenter to any vertex.
            # For the covering radius, we want the maximum circumradius over all facets.
            
            # Let me compute both and take the one that makes sense.
            # Actually, for a convex hull facet, the circumcenter of the spherical triangle
            # should be the one that is on the "inside" of the sphere, i.e., closer to the origin.
            # But both C1 and C2 are on the unit sphere.
            
            # Let me just compute the circumradius for both and take the maximum.
            # Actually, the covering radius is the maximum over all points on the sphere of the distance to the nearest configuration point.
            # For a facet, the farthest point from the three vertices is the circumcenter.
            # So the circumradius is the distance from the circumcenter to any vertex.
            
            # Let me compute cos(R) for both C1 and C2.
            # The circumcenter should satisfy cos(R) = C·p_i for all i.
            # For C1: cos(R1) = C1·p0, C1·p1, C1·p2
            # For C2: cos(R2) = C2·p0, C2·p1, C2·p2
            
            # Since C2 = -C1, cos(R2) = -cos(R1)
            
            # The circumradius R is arccos(cos(R))
            # We want the circumradius to be the angular distance, which is in [0, pi]
            
            # For the covering radius, we want the maximum circumradius.
            # Let me compute R1 = arccos(C1·p0) and R2 = arccos(C2·p0)
            # and take the maximum.
            
            R1 = math.acos(np.clip(C1[0] * facet[0][0] + C1[1] * facet[0][1] + C1[2] * facet[0][2], -1.0, 1.0))
            R2 = math.acos(np.clip(C2[0] * facet[0][0] + C2[1] * facet[0][1] + C2[2] * facet[0][2], -1.0, 1.0))
            
            # The circumcenter should be the one that is inside the spherical triangle
            # For a convex hull, the circumcenter of the facet is the one that is on the same side as the origin
            # Since n points outward, the origin is on the -n side, so the circumcenter should be C2 = -n/|n|
            
            # But let me verify by checking if C2·p_i is the same for all i
            dot0 = np.dot(C2, facet[0])
            dot1 = np.dot(C2, facet[1])
            dot2 = np.dot(C2, facet[2])
            
            if abs(dot0 - dot1) > 1e-6 or abs(dot0 - dot2) > 1e-6:
                # C2 is not the circumcenter, try C1
                dot0 = np.dot(C1, facet[0])
                dot1 = np.dot(C1, facet[1])
                dot2 = np.dot(C1, facet[2])
                if abs(dot0 - dot1) < 1e-6 and abs(dot0 - dot2) < 1e-6:
                    R = math.acos(np.clip(dot0, -1.0, 1.0))
                else:
                    # Numerical issue, use the average
                    R = math.acos(np.clip(np.mean([dot0, dot1, dot2]), -1.0, 1.0))
            else:
                R = math.acos(np.clip(dot0, -1.0, 1.0))
            
            max_circumradius = max(max_circumradius, math.degrees(R))
    
    return max_circumradius


def _refine_fibonacci(points, max_iter=100, tol=1e-8):
    """Refine a set of points on the unit sphere to minimize the covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_radius = _angular_distance_deg(best_points)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try moving point i in small random directions
            for _ in range(20):
                # Random direction
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                
                # Small step
                step = 0.01
                new_point = points[i] + step * direction
                new_point = new_point / np.linalg.norm(new_point)
                
                # Test
                test_points = points.copy()
                test_points[i] = new_point
                
                radius = _angular_distance_deg(test_points)
                if radius < best_radius - tol:
                    points = test_points
                    best_points = test_points.copy()
                    best_radius = radius
                    improved = True
        
        if not improved:
            break
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _angular_distance_deg(best_points)
    
    # Refine
    while time.time() < deadline:
        points, radius = _refine_fibonacci(best_points, max_iter=50)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        else:
            break
    
    return best_points
