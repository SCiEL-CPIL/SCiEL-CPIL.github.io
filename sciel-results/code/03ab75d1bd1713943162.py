"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import math

def _covering_radius(points):
    """Compute the covering radius (max circumradius of hull facets) in radians."""
    try:
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms < 1e-10):
            return np.pi  # invalid
        points = points / norms[:, None]
        
        # Check if all points are in a closed hemisphere
        # If so, the covering radius is pi (or we need to handle differently)
        # A set of points is in a closed hemisphere iff there exists a vector v
        # such that v . p_i >= 0 for all i. This is equivalent to 0 being in the convex hull
        # of the antipodal points, or more simply, we can check if the origin is inside
        # the convex hull of the points. If the origin is inside, they are NOT in a hemisphere.
        # Actually, if they ARE in a hemisphere, the origin is NOT in the convex hull of the points.
        # Wait, the condition for being in a closed hemisphere is that the convex hull does NOT contain the origin.
        
        # Let's just compute the hull. If the hull is degenerate or points are in a hemisphere,
        # the covering radius is effectively large.
        # For a proper covering, points should surround the origin.
        
        hull = ConvexHull(points)
        
        # For each facet, compute the circumradius
        # A facet is defined by 3 points (triangles) in 3D
        max_circumradius = 0.0
        
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumradius of triangle p1, p2, p3
            # Using the formula: R = abc / (4 * Area)
            # where a, b, c are side lengths and Area is the area of the triangle.
            
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            # Area using cross product
            cross = np.cross(p2 - p1, p3 - p1)
            area = 0.5 * np.linalg.norm(cross)
            
            if area < 1e-12:
                # Degenerate triangle, skip or handle
                continue
            
            # Circumradius
            R = (a * b * c) / (4.0 * area)
            
            # The circumradius in terms of angular distance from the facet center
            # The angular distance from the sphere to the nearest point on the facet
            # is related to the circumradius of the spherical triangle.
            # For a spherical triangle with side lengths a, b, c (angular distances),
            # the circumradius R satisfies:
            # cos(R) = (cos(a) + cos(b) + cos(c)) / (1 + 2*cos(a)*cos(b)*cos(c)) ... no that's not right.
            
            # Actually, the circumcenter of a spherical triangle is the point on the sphere
            # equidistant from all three vertices. The angular distance from the circumcenter
            # to each vertex is the circumradius.
            
            # Let's compute it directly. The circumcenter C of triangle p1, p2, p3 on the unit sphere
            # satisfies |C - p1| = |C - p2| = |C - p3| = 2*sin(R/2) where R is the angular circumradius.
            # Wait, |C - p_i| = 2*sin(R/2) where R is the angular distance.
            
            # The circumcenter C is in the plane of p1, p2, p3.
            # C = alpha * p1 + beta * p2 + gamma * p3
            # with alpha + beta + gamma = 1 (barycentric) and |C - p1| = |C - p2| = |C - p3|.
            
            # Alternatively, the circumcenter of a triangle in 3D (Euclidean) projected onto the sphere
            # gives the spherical circumcenter.
            
            # Let's compute the Euclidean circumcenter of the triangle p1, p2, p3.
            # The Euclidean circumcenter O_eucl satisfies |O_eucl - p1| = |O_eucl - p2| = |O_eucl - p3|.
            # O_eucl = p1 + s * (p2 - p1) + t * (p3 - p1)
            # |O_eucl - p1|^2 = |O_eucl - p2|^2
            # |s*(p2-p1) + t*(p3-p1)|^2 = |s*(p2-p1) + t*(p3-p1) - (p2-p1)|^2
            # Let u = p2 - p1, v = p3 - p1
            # |s*u + t*v|^2 = |(s-1)*u + t*v|^2
            # s^2|u|^2 + t^2|v|^2 + 2st u.v = (s-1)^2|u|^2 + t^2|v|^2 + 2t(s-1)u.v
            # s^2|u|^2 + 2st u.v = (s^2 - 2s + 1)|u|^2 + 2st u.v - 2t u.v
            # 0 = -2s|u|^2 + |u|^2 - 2t u.v
            # 2s|u|^2 + 2t u.v = |u|^2
            # s|u|^2 + t u.v = |u|^2 / 2  ... (1)
            
            # Similarly, |O_eucl - p1|^2 = |O_eucl - p3|^2
            # |s*u + t*v|^2 = |s*u + (t-1)*v|^2
            # s^2|u|^2 + t^2|v|^2 + 2st u.v = s^2|u|^2 + (t-1)^2|v|^2 + 2s(t-1)u.v
            # t^2|v|^2 + 2st u.v = (t^2 - 2t + 1)|v|^2 + 2st u.v - 2s u.v
            # 0 = -2t|v|^2 + |v|^2 - 2s u.v
            # 2t|v|^2 + 2s u.v = |v|^2
            # t|v|^2 + s u.v = |v|^2 / 2  ... (2)
            
            # Solve for s, t:
            # [ |u|^2   u.v  ] [s]   [ |u|^2 / 2 ]
            # [ u.v    |v|^2 ] [t] = [ |v|^2 / 2 ]
            
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-14:
                continue
            
            s = (vv * (uu / 2.0) - uv * (vv / 2.0)) / det
            t = (uu * (vv / 2.0) - uv * (uu / 2.0)) / det
            
            O_eucl = p1 + s * u + t * v
            
            # The Euclidean circumradius
            R_eucl = np.linalg.norm(O_eucl - p1)
            
            # The angular circumradius R_ang satisfies:
            # |C - p1| = 2 * sin(R_ang / 2) where C is the normalized circumcenter
            # But C = O_eucl / |O_eucl|
            # |C - p1| = |O_eucl/|O_eucl| - p1|
            # This is not simply related to R_eucl.
            
            # Actually, the angular distance from C to p1 is:
            # cos(R_ang) = C . p1 = (O_eucl . p1) / |O_eucl|
            
            O_eucl_norm = np.linalg.norm(O_eucl)
            if O_eucl_norm < 1e-14:
                continue
            
            C = O_eucl / O_eucl_norm
            cos_R = np.dot(C, p1)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R_ang = np.arccos(cos_R)
            
            if R_ang > max_circumradius:
                max_circumradius = R_ang
        
        return max_circumradius
    
    except Exception:
        return np.pi  # invalid configuration


def _refine(points, max_time=100.0):
    """Refine the points to minimize the covering radius."""
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius(current_points)
    
    # Use gradient-free optimization (Nelder-Mead) with a time limit
    # We'll use a simpler approach: iterative repulsion + local optimization
    
    # First, do a few rounds of repulsion to spread points out
    for iteration in range(50):
        if time.time() - start_time > max_time - 30:
            break
        
        # Compute repulsive forces
        forces = np.zeros_like(current_points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(current_points[i] - current_points[j])
                if d < 1e-10:
                    continue
                # Repulsive force proportional to 1/d^2
                force = (current_points[i] - current_points[j]) / (d * d)
                forces[i] += force
        
        # Move points in the direction of the force, projected onto the tangent plane
        for i in range(N):
            p = current_points[i]
            f = forces[i]
            # Project f onto the tangent plane at p
            f_tangent = f - np.dot(f, p) * p
            # Normalize and take a small step
            f_norm = np.linalg.norm(f_tangent)
            if f_norm > 1e-10:
                step_size = 0.01
                current_points[i] = p + step_size * f_tangent / f_norm
                # Re-normalize to unit sphere
                current_points[i] = current_points[i] / np.linalg.norm(current_points[i])
        
        new_radius = _covering_radius(current_points)
        if new_radius < current_radius:
            current_radius = new_radius
        else:
            # Revert if it got worse
            break
    
    # Now do a local optimization using Nelder-Mead
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None]
        return _covering_radius(pts)
    
    x0 = current_points.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if time.time() - start_time < max_time - 10:
            x_opt = result.x.reshape(N, 3)
            norms = np.linalg.norm(x_opt, axis=1)
            x_opt = x_opt / norms[:, None]
            opt_radius = _covering_radius(x_opt)
            
            if opt_radius < current_radius:
                current_points = x_opt
                current_radius = opt_radius
    except Exception:
        pass
    
    return current_points


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine the points
    points = _refine(points, max_time=110.0)
    
    return points
# EVOLVE-BLOCK-END
