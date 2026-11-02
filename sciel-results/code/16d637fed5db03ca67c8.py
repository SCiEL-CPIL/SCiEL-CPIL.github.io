import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points):
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_rad = 0.0
        for tri in facets:
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            # Compute circumcenter of triangle on the sphere
            # Using the formula for circumcenter of a triangle on the unit sphere
            # The circumcenter is proportional to a x (b x c) + b x (c x a) + c x (a x b)
            # But a simpler method is to solve the linear system for the center
            # Let's use the method of finding the center of the circle passing through a, b, c
            # This is equivalent to finding the point on the sphere that is equidistant from a, b, c
            # We can use the fact that the circumcenter is the intersection of the sphere with the line
            # through the centroid in the direction of the normal to the plane of the triangle
            
            # Normal to the plane
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            # The circumcenter lies on the line: centroid + t * n
            # We need to find t such that ||centroid + t*n|| = 1
            centroid = (a + b + c) / 3.0
            # ||centroid + t*n||^2 = 1
            # ||centroid||^2 + 2*t*(centroid.n) + t^2 = 1
            # t^2 + 2*(centroid.n)*t + (||centroid||^2 - 1) = 0
            a_coef = 1.0
            b_coef = 2.0 * np.dot(centroid, n)
            c_coef = np.dot(centroid, centroid) - 1.0
            
            disc = b_coef*b_coef - 4*a_coef*c_coef
            if disc < 0:
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b_coef + sqrt_disc) / (2*a_coef)
            t2 = (-b_coef - sqrt_disc) / (2*a_coef)
            
            # Choose the solution that is on the same side as the normal
            # The circumcenter should be in the direction of the normal from the centroid
            # Actually, the circumcenter is the point on the sphere closest to the centroid
            # in the direction of the normal? No, it's the point on the sphere that is equidistant.
            # Let's check both solutions and pick the one that is on the sphere and
            # such that the distance to the vertices is minimized? No, it's the circumradius.
            # The circumradius is the distance from the circumcenter to any vertex.
            # We want the largest circumradius over all facets.
            
            # Let's just compute the circumradius directly using the formula
            # R = abc / (4 * Area)
            # where a, b, c are the side lengths and Area is the area of the triangle
            # on the sphere. But this is for the spherical triangle.
            # For the Euclidean triangle formed by the points on the sphere, the circumradius
            # of the Euclidean triangle is not the same as the spherical circumradius.
            
            # Let's use the spherical distance.
            # The covering radius is the maximum over all points y on the sphere of
            # min_i ||y - p_i||. This is equal to the maximum circumradius of the
            # spherical triangles formed by the facets of the convex hull.
            
            # A simpler approach: the circumcenter of the spherical triangle is the
            # point on the sphere that is equidistant from the three vertices.
            # We can find it by solving the system:
            # x.a = x.b = x.c = cos(theta)
            # x.x = 1
            # This is a linear system if we fix theta.
            
            # Let's use the property that the circumcenter is the intersection of the
            # sphere with the line through the centroid in the direction of the normal.
            # We already have the line. Let's pick the correct t.
            # The circumcenter should be on the same side of the triangle as the normal.
            # Let's check the dot product of (centroid + t*n) - centroid with n.
            # It should be positive for the correct solution.
            
            # Actually, let's just compute both and pick the one with larger t if n is outward.
            # The normal n is outward if it points away from the origin.
            # Since the points are on the unit sphere, the convex hull contains the origin.
            # The normal to the facet points outward.
            
            # Let's pick t1 if t1 > 0 else t2.
            if t1 > 0:
                t = t1
            else:
                t = t2
            
            center = centroid + t * n
            center = center / np.linalg.norm(center)
            
            # Compute the angular distance from center to one of the vertices
            dot = np.dot(center, a)
            dot = np.clip(dot, -1.0, 1.0)
            rad = np.arccos(dot)
            if rad > max_rad:
                max_rad = rad
        return max_rad
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    deadline = 150.0
    
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_rad = _covering_radius(points)
    
    # Optimization loop
    step_size = 0.05
    n_iter = 0
    while time.time() < deadline - 5.0:
        n_iter += 1
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
        
        # Compute forces on each point
        forces = np.zeros_like(points)
        for tri in facets:
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            centroid = (a + b + c) / 3.0
            a_coef = 1.0
            b_coef = 2.0 * np.dot(centroid, n)
            c_coef = np.dot(centroid, centroid) - 1.0
            
            disc = b_coef*b_coef - 4*a_coef*c_coef
            if disc < 0:
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b_coef + sqrt_disc) / (2*a_coef)
            t2 = (-b_coef - sqrt_disc) / (2*a_coef)
            
            if t1 > 0:
                t = t1
            else:
                t = t2
            
            center = centroid + t * n
            center = center / np.linalg.norm(center)
            
            dot = np.dot(center, a)
            dot = np.clip(dot, -1.0, 1.0)
            rad = np.arccos(dot)
            
            # Move points away from the center proportional to the radius
            # The force on each vertex is in the direction from the center to the vertex
            for idx in tri:
                v = points[idx]
                force = v - center
                force = force / np.linalg.norm(force)
                forces[idx] += force * rad
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update points
        points = points + step_size * forces
        points = _normalize(points)
        
        # Check for improvement
        rad = _covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
        
        # Adaptive step size
        if rad > best_rad:
            step_size *= 0.9
        else:
            step_size = min(step_size * 1.05, 0.1)
    
    # Final polish
    if time.time() < deadline - 1.0:
        points = best_points.copy()
        step_size = 0.01
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            try:
                hull = ConvexHull(points)
                facets = hull.simplices
            except Exception:
                break
            
            forces = np.zeros_like(points)
            for tri in facets:
                a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                centroid = (a + b + c) / 3.0
                a_coef = 1.0
                b_coef = 2.0 * np.dot(centroid, n)
                c_coef = np.dot(centroid, centroid) - 1.0
                
                disc = b_coef*b_coef - 4*a_coef*c_coef
                if disc < 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = (-b_coef + sqrt_disc) / (2*a_coef)
                t2 = (-b_coef - sqrt_disc) / (2*a_coef)
                
                if t1 > 0:
                    t = t1
                else:
                    t = t2
                
                center = centroid + t * n
                center = center / np.linalg.norm(center)
                
                dot = np.dot(center, a)
                dot = np.clip(dot, -1.0, 1.0)
                rad = np.arccos(dot)
                
                for idx in tri:
                    v = points[idx]
                    force = v - center
                    force = force / np.linalg.norm(force)
                    forces[idx] += force * rad
            
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            forces = forces / force_norms
            
            points = points + step_size * forces
            points = _normalize(points)
            
            rad = _covering_radius(points)
            if rad < best_rad:
                best_rad = rad
                best_points = points.copy()
    
    return best_points
