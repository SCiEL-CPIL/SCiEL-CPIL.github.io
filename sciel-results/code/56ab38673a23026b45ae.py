"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import SphericalVoronoi

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def _compute_covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    try:
        # Compute convex hull
        hull = ConvexHull(points)
        
        # Get the facets (triangles) of the convex hull
        simplices = hull.simplices
        
        # For each facet, compute the circumradius (angular radius of the empty cap)
        max_radius = 0.0
        
        for simplex in simplices:
            # Get the three points of the triangle
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Compute the center of the circumcircle (in 3D, this is the point equidistant from all three)
            # For a spherical triangle, the circumcenter is in the direction of the cross product
            # Actually, for points on a unit sphere, the circumradius of the spherical triangle
            # can be computed from the angles between the points
            
            # Compute the three side lengths (angular distances)
            def angular_dist(a, b):
                dot = np.dot(a, b)
                dot = np.clip(dot, -1.0, 1.0)
                return np.arccos(dot)
            
            a = angular_dist(p2, p3)
            b = angular_dist(p1, p3)
            c = angular_dist(p1, p2)
            
            # Use the spherical law of cosines to find the circumradius
            # For a spherical triangle with sides a, b, c, the circumradius R satisfies:
            # cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
            # where A is the angle opposite to side a
            # The circumradius can be found using:
            # sin(R) = sin(a)sin(b)sin(c) / (2 * cos(s/2) * cos((s-a)/2) * cos((s-b)/2) * cos((s-c)/2))
            # where s = (a+b+c)/2
            # But this is complex. A simpler approach:
            # The circumcenter of the triangle on the sphere is the intersection of the perpendicular bisectors
            # Alternatively, we can compute the circumcenter in 3D space
            
            # Compute the circumcenter of the triangle in 3D
            # The circumcenter is equidistant from p1, p2, p3
            # Let center = alpha * p1 + beta * p2 + gamma * p3
            # We want |center - p1| = |center - p2| = |center - p3|
            
            # A simpler approach: the circumcenter lies in the plane of the triangle
            # and is perpendicular to the plane's normal
            # Actually, for points on a unit sphere, the circumcenter of the spherical triangle
            # is in the direction of the vector sum of the three points (approximately)
            
            # Let's use a different approach: compute the circumcenter by solving the system
            # The circumcenter C satisfies:
            # |C - p1|^2 = |C - p2|^2 = |C - p3|^2
            # This gives us two linear equations
            # C . (p2 - p1) = (|p2|^2 - |p1|^2) / 2 = 0 (since |p1|=|p2|=1)
            # C . (p3 - p1) = 0
            # So C is perpendicular to (p2-p1) and (p3-p1)
            # Thus C is in the direction of the cross product (p2-p1) x (p3-p1)
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-10:
                # Degenerate triangle, skip
                continue
            
            # The circumcenter is in the direction of normal, but we need to find the correct sign
            # and scale
            # Actually, the circumcenter of the triangle on the sphere is the point on the sphere
            # that is equidistant from p1, p2, p3
            # This is the intersection of the sphere with the line through the origin in the direction
            # of the circumcenter of the planar triangle
            
            # Compute the circumcenter of the planar triangle formed by p1, p2, p3
            # The circumcenter in 3D space (not necessarily on the unit sphere) can be found by:
            # Let A = p1, B = p2, C = p3
            # d = 2 * (A_x * (B_y * C_z - C_y * B_z) + B_x * (C_y * A_z - A_y * C_z) + C_x * (A_y * B_z - B_y * A_z))
            # Ux = (|B|^2 * (A_y * C_z - A_z * C_y) + |C|^2 * (A_z * B_y - A_y * B_z)) / d
            # Uy = (|A|^2 * (B_z * C_x - B_x * C_z) + |C|^2 * (B_x * A_z - B_z * A_x)) / d
            # Uz = (|A|^2 * (B_x * C_y - B_y * C_x) + |B|^2 * (B_y * A_x - B_x * A_y)) / d
            
            Ax, Ay, Az = p1
            Bx, By, Bz = p2
            Cx, Cy, Cz = p3
            
            A2 = Ax**2 + Ay**2 + Az**2  # = 1
            B2 = Bx**2 + By**2 + Bz**2  # = 1
            C2 = Cx**2 + Cy**2 + Cz**2  # = 1
            
            d = 2 * (Ax * (By * Cz - Cy * Bz) + Bx * (Cy * Az - Ay * Cz) + Cx * (Ay * Bz - By * Az))
            
            if abs(d) < 1e-10:
                # Degenerate, skip
                continue
            
            Ux = (B2 * (Ay * Cz - Az * Cy) + C2 * (Az * By - Ay * Bz)) / d
            Uy = (A2 * (Bz * Cx - Bx * Cz) + C2 * (Bx * Az - Bz * Ax)) / d
            Uz = (A2 * (Bx * Cy - By * Cx) + B2 * (By * Ax - Bx * Ay)) / d
            
            circumcenter = np.array([Ux, Uy, Uz])
            circumcenter_norm = np.linalg.norm(circumcenter)
            
            if circumcenter_norm < 1e-10:
                continue
            
            circumcenter_unit = circumcenter / circumcenter_norm
            
            # The circumradius is the angular distance from the circumcenter to any of the three points
            dot = np.dot(circumcenter_unit, p1)
            dot = np.clip(dot, -1.0, 1.0)
            radius = np.arccos(dot)
            
            # Check if this is the correct circumcenter (should be on the same side as the triangle)
            # The circumcenter should be such that the cap is empty
            # For now, just take the radius
            if radius > max_radius:
                max_radius = radius
        
        return np.degrees(max_radius)
    
    except Exception:
        return np.pi  # Return a large value on error

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N).copy()
    
    # Normalize
    for i in range(N):
        points[i] = _normalize(points[i])
    
    # Time limit: 150 seconds total, leave 10 seconds for safety
    start_time = time.time()
    time_limit = 140.0
    
    # Iterative refinement
    # We'll use a simple local optimization: for each point, try to move it
    # to reduce the maximum circumradius
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Use a more sophisticated approach: gradient descent on the covering radius
    # But since the covering radius is not differentiable, we'll use a stochastic approach
    
    # Actually, let's use a simpler but effective approach:
    # 1. Compute the convex hull and find the largest facet
    # 2. Move the vertices of that facet apart slightly
    # 3. Repeat
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
            
            # Find the facet with the largest circumradius
            max_radius = 0
            max_facet = None
            
            for simplex in simplices:
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                # Compute circumradius
                Ax, Ay, Az = p1
                Bx, By, Bz = p2
                Cx, Cy, Cz = p3
                
                A2 = 1.0
                B2 = 1.0
                C2 = 1.0
                
                d = 2 * (Ax * (By * Cz - Cy * Bz) + Bx * (Cy * Az - Ay * Cz) + Cx * (Ay * Bz - By * Az))
                
                if abs(d) < 1e-10:
                    continue
                
                Ux = (B2 * (Ay * Cz - Az * Cy) + C2 * (Az * By - Ay * Bz)) / d
                Uy = (A2 * (Bz * Cx - Bx * Cz) + C2 * (Bx * Az - Bz * Ax)) / d
                Uz = (A2 * (Bx * Cy - By * Cx) + B2 * (By * Ax - Bx * Ay)) / d
                
                circumcenter = np.array([Ux, Uy, Uz])
                circumcenter_norm = np.linalg.norm(circumcenter)
                
                if circumcenter_norm < 1e-10:
                    continue
                
                circumcenter_unit = circumcenter / circumcenter_norm
                dot = np.dot(circumcenter_unit, p1)
                dot = np.clip(dot, -1.0, 1.0)
                radius = np.arccos(dot)
                
                if radius > max_radius:
                    max_radius = radius
                    max_facet = simplex
            
            if max_facet is None:
                break
            
            # Move the vertices of the largest facet apart
            # This is a heuristic: push each vertex away from the circumcenter
            circumcenter = np.array([
                (B2 * (Ay * Cz - Az * Cy) + C2 * (Az * By - Ay * Bz)) / d,
                (A2 * (Bz * Cx - Bx * Cz) + C2 * (Bx * Az - Bz * Ax)) / d,
                (A2 * (Bx * Cy - By * Cx) + B2 * (By * Ax - Bx * Ay)) / d
            ])
            circumcenter_norm = np.linalg.norm(circumcenter)
            if circumcenter_norm < 1e-10:
                continue
            circumcenter_unit = circumcenter / circumcenter_norm
            
            # Push each vertex away from the circumcenter
            step_size = 0.01
            for idx in max_facet:
                direction = points[idx] - circumcenter_unit
                direction = _normalize(direction)
                new_point = points[idx] + step_size * direction
                new_point = _normalize(new_point)
                points[idx] = new_point
            
            # Compute new radius
            new_radius = _compute_covering_radius(points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        
        except Exception:
            continue
    
    # Return the best configuration found
    return best_points
# EVOLVE-BLOCK-END
