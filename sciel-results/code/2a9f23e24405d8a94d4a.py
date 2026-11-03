import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    def objective(x):
        # Check time
        if time.time() > deadline:
            raise TimeoutError()
            
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute Delaunay triangulation
        # Note: Delaunay on 3D points gives tetrahedra if 3D, but we are on sphere.
        # Actually, scipy.spatial.Delaunay works on 3D points.
        # The facets of the convex hull of the points on the sphere correspond to the spherical Delaunay triangles.
        # We need the facets of the convex hull, not the tetrahedra inside.
        # scipy.spatial.ConvexHull gives the facets.
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If points are coplanar or degenerate, return large penalty
            return 1e6
            
        # hull.simplices contains the indices of the points forming each facet (triangle)
        # For each facet, compute the circumradius on the sphere
        
        max_radius = 0.0
        sum_sq_radius = 0.0
        
        for simplex in hull.simplices:
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute circumcenter of the spherical triangle
            # The circumcenter is the intersection of the sphere with the plane perpendicular to the triangle's plane passing through the origin?
            # No, the circumcenter of a spherical triangle is the point on the sphere equidistant from the three vertices.
            # This is the normalization of the vector cross product? No.
            # The circumcenter is in the direction of the normal to the plane defined by p1, p2, p3?
            # Let's check. The plane equation is n . x = d. Since origin is inside, d is small?
            # Actually, the circumcenter of the triangle on the sphere is the point c on the sphere such that c.p1 = c.p2 = c.p3.
            # This implies c is orthogonal to (p1-p2) and (p1-p3).
            # So c is parallel to (p1-p2) x (p1-p3).
            
            v1 = p1 - p2
            v2 = p1 - p3
            c_vec = np.cross(v1, v2)
            
            # Check if c_vec is zero (degenerate triangle)
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-10:
                # Degenerate, use midpoint or something
                c = (p1 + p2 + p3) / 3.0
                c = c / np.linalg.norm(c)
            else:
                c = c_vec / c_norm
            
            # The circumradius is the angular distance from c to p1
            # cos(theta) = c . p1
            dot = np.dot(c, p1)
            # Clamp for numerical stability
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            
            sum_sq_radius += theta * theta
            if theta > max_radius:
                max_radius = theta
                
        # Return sum of squares to encourage uniformity, or max to minimize worst case.
        # Sum of squares is smoother.
        return sum_sq_radius

    # Use L-BFGS-B
    # We can try to optimize the sum of squares.
    
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10, 'maxfun': 5000})
    
    if time.time() > deadline:
         # Return best found so far? minimize doesn't give partial results easily.
         # We should probably just return the initial or last known good state if it times out.
         # But minimize raises exception? No, it just stops.
         # We need to handle the case where it runs out of time.
         pass

    x_opt = res.x
    pts_opt = x_opt.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / norms
    
    return pts_opt
