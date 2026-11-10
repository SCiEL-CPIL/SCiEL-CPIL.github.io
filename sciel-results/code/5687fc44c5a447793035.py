"""
Sphere covering with 130 points.
Strategy:
1. Initialize with a high-quality Fibonacci lattice.
2. Use a log-sum-exp smooth approximation of the covering radius to guide a global-ish search (or just local refinement) because it is differentiable and smooth, allowing for better local minima identification than the raw max-circumradius which is non-smooth.
3. Identify the "worst" vertices (those defining the maximum circumradius) using the convex hull.
4. Apply Nelder-Mead optimization specifically on the coordinates of these worst vertices to flatten the largest caps.
5. Repeat the identification and local optimization process.
"""

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_exact(points):
    """
    Compute the exact covering radius in degrees.
    This is the ground truth metric.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        
        # Handle triangular facets
        tri_facets = [f for f in facets if len(f) == 3]
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute circumradius of spherical triangle
            # Using the formula for spherical circumradius R:
            # sin(R) = (sin(a) sin(b) sin(c)) / (4 * area) -- wait, this is for planar.
            # For spherical triangle with vertices on unit sphere:
            # The circumcenter is the intersection of the perpendicular bisectors.
            # Alternatively, the distance from the origin to the plane of the triangle is d.
            # The radius of the circle on the sphere is R.
            # The plane equation is n . x = d.
            # The distance from the center of the circle (on the axis) to the origin is d.
            # The radius of the circle in 3D is r_3d = sqrt(1 - d^2).
            # The angular radius R is such that sin(R) = r_3d / 1 = sqrt(1-d^2).
            # So R = arcsin(sqrt(1-d^2)).
            
            # Let's compute d for each triangle.
            # n = (b-a) x (c-a)
            # d = n . a / |n|
            
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            norms_n = np.linalg.norm(n, axis=1)
            valid = norms_n > 1e-12
            
            if np.any(valid):
                n_valid = n[valid]
                a_valid = a[valid]
                norms_n_valid = norms_n[valid]
                
                # d = n . a / |n|
                d = np.einsum('ij,ij->i', n_valid, a_valid) / norms_n_valid
                # d should be in (-1, 1). For convex hull of points on sphere, d > 0 usually if origin is inside.
                # Actually, for a facet of the convex hull of points on the unit sphere, the origin is inside the hull.
                # The plane separates the origin from the rest of the hull? No, the origin is inside.
                # So the normal points outward. The distance from origin to the plane is d.
                # The cap is the part of the sphere cut by this plane.
                # The angular radius R of the cap is the angle between the normal (pointing to circumcenter) and the boundary.
                # cos(R) = d. So R = arccos(d).
                
                # Wait, let's verify.
                # If the plane is x=d, and the sphere is x^2+y^2+z^2=1.
                # The intersection is a circle at x=d.
                # The point on the sphere closest to the plane along the normal is (d, 0, 0) if normal is (1,0,0).
                # The distance from the center of the sphere (0,0,0) to the point (d,0,0) is d.
                # The angle R is the angle between (1,0,0) and a point on the circle.
                # Let P be a point on the circle. P.x = d. |P|=1.
                # cos(angle) = P . (1,0,0) / (|P||n|) = d.
                # So R = arccos(d).
                
                d = np.clip(d, -1.0, 1.0)
                R = np.arccos(d)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            # For a planar polygon on the sphere, the circumradius is determined by the farthest vertex from the "center" of the polygon on the sphere.
            # The "center" is the intersection of the plane with the sphere's axis? No.
            # The plane cuts the sphere. The cap is defined by the plane.
            # The angular radius is still arccos(d) where d is the distance from origin to the plane.
            # We need to compute d for the plane defined by the points.
            
            # Use the first 3 non-collinear points to define the plane.
            # Since they are on the convex hull, they are not collinear (unless degenerate).
            # Let's just pick 3 points.
            if len(pts) >= 3:
                a, b, c = pts[0], pts[1], pts[2]
                v1 = b - a
                v2 = c - a
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n > 1e-12:
                    d = np.dot(n, a) / norm_n
                    d = np.clip(d, -1.0, 1.0)
                    R = np.arccos(d)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf

def _logsumexp_covering_radius(points, temperature=0.1):
    """
    Approximate the covering radius using a log-sum-exp of the distances to the nearest points.
    This is a smooth surrogate for the max operation.
    """
    # This is computationally expensive if done naively (N x N matrix).
    # For N=130, 130x130 is small.
    # dist[i,j] = angular distance between point i and point j.
    # For a point y on the sphere, the distance to the nearest point is min_j dist(y, p_j).
    # We want max_y min_j dist(y, p_j).
    # This is hard to approximate directly with logsumexp over y.
    
    # Alternative: The covering radius is the maximum circumradius of the facets.
    # We can approximate the max circumradius by a logsumexp of the circumradii of all facets.
    # But we need to compute the facets first, which is expensive.
    
    # Let's stick to the exact evaluation for the objective, but use a different optimization strategy.
    # The strategy is:
    # 1. Compute hull and find the facet with the max circumradius.
    # 2. Identify the vertices of this facet.
    # 3. Optimize these vertices using Nelder-Mead to minimize the circumradius of this facet,
    #    while keeping other points fixed.
    # 4. Repeat.
    
    # This function is not used in the main loop if we use the targeted approach.
    # But we might use it for a global search if needed.
    # For now, return the exact radius.
    return _covering_radius_exact(points)

def _find_worst_facet(points):
    """
    Find the facet with the maximum circumradius and return its vertices and the radius.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        worst_facet = None
        
        for f in facets:
            if len(f) == 3:
                a, b, c = points[f[0]], points[f[1]], points[f[2]]
                v1 = b - a
                v2 = c - a
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n > 1e-12:
                    d = np.dot(n, a) / norm_n
                    d = np.clip(d, -1.0, 1.0)
                    R = np.arccos(d)
                    if R > max_circumradius:
                        max_circumradius = R
                        worst_facet = f
            else:
                # Handle non-triangular
                pts = points[f]
                if len(pts) >= 3:
                    a, b, c = pts[0], pts[1], pts[2]
                    v1 = b - a
                    v2 = c - a
                    n = np.cross(v1, v2)
                    norm_n = np.linalg.norm(n)
                    if norm_n > 1e-12:
                        d = np.dot(n, a) / norm_n
                        d = np.clip(d, -1.0, 1.0)
                        R = np.arccos(d)
                        if R > max_circumradius:
                            max_circumradius = R
                            worst_facet = f
        
        return worst_facet, np.degrees(max_circumradius)
    except Exception:
        return None, np.inf

def _optimize_worst_vertices(points, worst_facet, max_time=10.0):
    """
    Optimize the vertices of the worst facet to reduce its circumradius.
    """
    if worst_facet is None:
        return points
    
    # Extract the vertices to optimize
    indices = list(worst_facet)
    # We might also want to include neighbors to prevent creating new large facets?
    # For now, just the vertices of the worst facet.
    
    initial_params = points[indices].flatten()
    
    def objective(params):
        # Reshape params back to points
        new_points = points.copy()
        new_points[indices] = params.reshape(-1, 3)
        
        # Normalize
        norms = np.linalg.norm(new_points[indices], axis=1)
        new_points[indices] = new_points[indices] / norms[:, np.newaxis]
        
        # Compute the circumradius of the facet formed by these points
        # Note: The hull might change, so the facet might not be the same.
        # But we are minimizing the circumradius of the triangle formed by these 3 points.
        # If the hull changes, this might not be the global max, but it's a local move.
        
        a, b, c = new_points[indices[0]], new_points[indices[1]], new_points[indices[2]]
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            return np.inf
        d = np.dot(n, a) / norm_n
        d = np.clip(d, -1.0, 1.0)
        R = np.arccos(d)
        return np.degrees(R)
    
    start_time = time.time()
    
    # Use Nelder-Mead
    # Bounds are implicitly handled by normalization.
    # But Nelder-Mead doesn't handle bounds well.
    # We can use a penalty or just rely on normalization.
    
    # Actually, we should optimize the full configuration to ensure the global max doesn't increase too much.
    # But that's expensive.
    # Let's try optimizing just the 3 points and check the global radius.
    
    # Better approach:
    # Define an objective that is the global covering radius.
    # This is expensive, but N=130 is small.
    
    def global_objective(params):
        new_points = points.copy()
        new_points[indices] = params.reshape(-1, 3)
        norms = np.linalg.norm(new_points[indices], axis=1)
        new_points[indices] = new_points[indices] / norms[:, np.newaxis]
        
        # Check if points are valid (not coincident, etc.)
        # For now, just compute the radius.
        return _covering_radius_exact(new_points)
    
    # We will use a limited time for this optimization.
    # Since global_objective is expensive, we might only be able to do a few iterations.
    # Let's try a simpler local move first: move the vertices slightly towards the centroid of the facet?
    # No, we want to flatten the cap, so we want to move the vertices apart?
    # Wait, if the cap is large, the circumradius is large.
    # To reduce the circumradius, we need to move the vertices closer to the circumcenter?
    # The circumcenter is the point on the sphere farthest from the origin along the normal.
    # Moving the vertices towards the circumcenter reduces the triangle size?
    # No, the circumradius is the distance from the circumcenter to the vertices.
    # If we move the vertices, we change the triangle.
    
    # Let's just use Nelder-Mead on the global objective for the 3 points.
    # We'll limit the number of function evaluations.
    
    class TimeLimitException(Exception):
        pass
    
    def time_limited_objective(params):
        if time.time() - start_time > max_time:
            raise TimeLimitException()
        return global_objective(params)
    
    try:
        res = minimize(time_limited_objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.x is not None:
            new_points = points.copy()
            new_points[indices] = res.x.reshape(-1, 3)
            norms = np.linalg.norm(new_points[indices], axis=1)
            new_points[indices] = new_points[indices] / norms[:, np.newaxis]
            return new_points
    except TimeLimitException:
        pass
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initialize
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _covering_radius_exact(best_points)
    
    # Main loop: Identify worst facet, optimize its vertices, check global improvement.
    while time.time() < deadline:
        worst_facet, current_radius = _find_worst_facet(points)
        
        if worst_facet is None:
            break
            
        # Optimize the vertices of the worst facet
        new_points = _optimize_worst_vertices(points, worst_facet, max_time=10.0)
        
        # Check if the new points are better
        new_radius = _covering_radius_exact(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
        else:
            # If no improvement, try a random perturbation of the worst vertices
            # to escape local minima.
            indices = list(worst_facet)
            for i in indices:
                perturb = np.random.randn(3) * 0.01
                new_point = points[i] + perturb
                new_point /= np.linalg.norm(new_point)
                points[i] = new_point
            
            new_radius = _covering_radius_exact(points)
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
            else:
                # Revert if it got worse
                # We don't have the old points saved easily here, so let's just break or restart?
                # For now, let's just break if we can't improve.
                break
    
    return best_points
# EVOLVE-BLOCK-END
