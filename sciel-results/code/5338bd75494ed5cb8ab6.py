import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # CVT Optimization
    # We will run a number of iterations to refine the points
    # The covering radius is the max circumradius of the convex hull facets
    
    def compute_circumcenters(points):
        """
        Compute circumcenters and circumradii for the convex hull facets.
        Returns:
            centers: array of shape (num_facets, 3)
            radii: array of shape (num_facets,)
            simplices: array of shape (num_facets, 3)
        """
        hull = scipy.spatial.ConvexHull(points)
        simplices = hull.simplices
        num_facets = len(simplices)
        centers = np.zeros((num_facets, 3))
        radii = np.zeros(num_facets)
        
        for idx, simplex in enumerate(simplices):
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Circumcenter of a triangle on the sphere
            # The circumcenter is the intersection of the sphere with the plane defined by p1, p2, p3
            # Normal to the plane
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                # Degenerate triangle, skip or handle
                centers[idx] = (p1 + p2 + p3) / 3.0
                radii[idx] = 0.0
                continue
            n = n / n_norm
            
            # Distance from origin to the plane
            # d = n . p1
            d = np.dot(n, p1)
            
            # The circumcenter C satisfies |C|^2 = 1 and C . n = d
            # C = d*n + v, where v is perpendicular to n
            # Actually, the circumcenter is the point on the line d*n + t*(n x n_perp) ...
            # Simpler: The circumcenter is the normalized vector (p1 + p2 + p3) only if the triangle is equilateral? No.
            # The circumcenter of the triangle on the sphere is the point C on the sphere such that C is equidistant from p1, p2, p3.
            # This is the intersection of the sphere and the plane n.x = d.
            # The vector from the origin to the plane is d*n.
            # The circumcenter is the point on the sphere closest to the plane's normal direction?
            # No, the circumcenter is the point C on the sphere such that C lies in the plane.
            # C = d*n + sqrt(1 - d^2) * u, where u is a unit vector perpendicular to n.
            # But there are two such points. The circumcenter is the one that is the "center" of the arc.
            # Actually, for a triangle on the unit sphere, the circumcenter is simply the normalized sum of the vertices?
            # Let's check. If p1, p2, p3 are on the unit sphere, is (p1+p2+p3) the circumcenter?
            # Only if the triangle is equilateral.
            
            # Correct method:
            # The circumcenter C is the solution to:
            # |C|^2 = 1
            # (C - p1) . n = 0 => C . n = p1 . n = d
            # We need another constraint? No, the intersection of a sphere and a plane is a circle.
            # The circumcenter of the triangle is the center of this circle? No.
            # The circumcenter of the triangle is the point on the sphere that is equidistant from p1, p2, p3.
            # This point is the intersection of the sphere and the plane.
            # Wait, any point on the circle of intersection is equidistant from p1, p2, p3?
            # No, the distance from C to p1 is sqrt(|C|^2 + |p1|^2 - 2 C.p1) = sqrt(2 - 2 C.p1).
            # So we need C.p1 = C.p2 = C.p3.
            # This implies C is in the plane defined by p1, p2, p3.
            # And C is on the sphere.
            # The locus of points on the sphere equidistant from p1 and p2 is the great circle perpendicular to the chord p1p2.
            # The intersection of the three great circles is the circumcenter.
            
            # Let's use the formula:
            # C is proportional to (p1 + p2 + p3) only if the origin is the circumcenter, which it is not.
            # The circumcenter is the normalized vector:
            # C = normalize( (p1 + p2 + p3) + k * n )?
            
            # Let's solve for C directly.
            # C = d*n + s * u, where u is a unit vector in the plane perpendicular to n.
            # We need |C|^2 = 1 => d^2 + s^2 = 1 => s = sqrt(1 - d^2).
            # We need to find the correct direction u.
            # The circumcenter must be equidistant from p1, p2, p3.
            # This is automatically satisfied for any point on the circle of intersection?
            # No. The distance from C to p1 is the chord length.
            # The chord length is determined by the angle.
            # The circumradius R is the angle between C and p1.
            # cos(R) = C.p1.
            # Since C is on the circle, C.p1 is constant for all C on the circle?
            # C = d*n + s*u.
            # C.p1 = d*(n.p1) + s*(u.p1) = d*d + s*(u.p1) = d^2 + s*(u.p1).
            # This is not constant unless u.p1 is constant, which it isn't.
            
            # I made a mistake. The circumcenter of a spherical triangle is NOT any point on the circle of intersection.
            # It is the unique point on the sphere that is equidistant from the three vertices.
            # This point is the intersection of the sphere and the plane? No, that's a circle.
            # The condition is C.p1 = C.p2 = C.p3.
            # This implies C is in the plane p1, p2, p3.
            # So C is on the circle.
            # But we need C.p1 = C.p2.
            # C.p1 - C.p2 = C.(p1 - p2) = 0.
            # So C is also in the plane defined by the origin and the vector p1-p2?
            # No, C.(p1-p2) = 0 means C is perpendicular to p1-p2.
            # So C is the intersection of the sphere, the plane of the triangle, and the plane perpendicular to p1-p2 passing through the origin?
            # No, the plane perpendicular to p1-p2 passing through the origin is the plane where points are equidistant from p1 and p2.
            # So C is the intersection of three planes:
            # 1. n.C = d (Plane of the triangle)
            # 2. (p1-p2).C = 0 (Equidistant from p1, p2)
            # 3. (p1-p3).C = 0 (Equidistant from p1, p3)
            # And |C| = 1.
            
            # Let's solve the linear system for C:
            # [ n       ] [Cx]   [d ]
            # [ p1-p2   ] [Cy] = [0 ]
            # [ p1-p3   ] [Cz]   [0 ]
            
            A = np.array([n, p1 - p2, p1 - p3])
            b = np.array([d, 0, 0])
            
            try:
                C = np.linalg.solve(A, b)
                C_norm = np.linalg.norm(C)
                C = C / C_norm
                # Calculate radius
                # cos(R) = C.p1
                cos_R = np.dot(C, p1)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                centers[idx] = C
                radii[idx] = R
            except np.linalg.LinAlgError:
                # Degenerate case
                centers[idx] = (p1 + p2 + p3) / 3.0
                centers[idx] /= np.linalg.norm(centers[idx])
                radii[idx] = 0.0
        
        return centers, radii, simplices

    # Optimization loop
    alpha = 0.1 # Learning rate
    for _ in range(500):
        centers, radii, simplices = compute_circumcenters(points)
        
        # For each point, accumulate the target position
        # We want to move point i towards the circumcenters of the triangles it belongs to
        # Weight by 1/radius to prioritize fixing large holes
        targets = np.zeros_like(points)
        weights = np.zeros(N)
        
        for idx, simplex in enumerate(simplices):
            R = radii[idx]
            if R < 1e-6:
                continue
            w = 1.0 / R
            C = centers[idx]
            for k in range(3):
                i = simplex[k]
                targets[i] += w * C
                weights[i] += w
        
        # Update points
        for i in range(N):
            if weights[i] > 0:
                target = targets[i] / weights[i]
                # Move towards target
                points[i] = points[i] + alpha * (target - points[i])
                # Normalize
                norm = np.linalg.norm(points[i])
                points[i] /= norm
        
    return points
