"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: dense grid evaluation
        # Create a reasonably dense grid
        n_grid = 200
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2*np.pi, 2*n_grid, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Compute min angular distance from each grid point to any site
        # Using dot product: cos(angle) = dot(y, p)
        # We want max over y of min over p of arccos(dot(y,p))
        # Which is equivalent to min over p of (1 - dot) for small angles, but let's use arccos
        # For efficiency, compute min dot for each grid point
        dots = grid_pts @ pts.T  # (n_grid_pts, N)
        min_dots = np.min(dots, axis=1)
        # Clamp to [-1, 1] for numerical safety
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        return np.max(angles) * 180.0 / np.pi
    
    # For each facet, find the circumradius
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # The circumcenter of a spherical triangle (facet) is the direction of the sum of the vertices
        # if the triangle is small, or more generally, we need to find the point on the sphere
        # that is equidistant from all three vertices and maximizes the distance.
        # For a convex hull on the unit sphere, the "worst" point for a facet is the point
        # on the great circle arc opposite to the centroid, but actually it's the circumcenter.
        
        # A more robust approach: the covering radius for a facet is the maximum angular distance
        # from any point in the spherical triangle to its vertices. For a spherical triangle,
        # this is either the circumradius (if the circumcenter is inside the triangle) or the
        # maximum of the angular distances from the circumcenter to the vertices... 
        # Actually, the farthest point in a spherical triangle from the vertices is the circumcenter
        # if it lies within the triangle, otherwise it's one of the vertices (but that would be 0).
        # Wait, the covering radius is the max over the whole sphere. For a facet, the points
        # that are NOT covered by any other facet's Voronoi cell are exactly this facet's region.
        # The worst point in this region is the circumcenter of the spherical triangle.
        
        # Let's compute the circumcenter of the spherical triangle defined by the 3 points.
        # The circumcenter is the point c such that |c - p1| = |c - p2| = |c - p3| (angular distance).
        # This is equivalent to c being orthogonal to the plane of the triangle in a certain sense?
        # No, for spherical geometry, the circumcenter is the intersection of the perpendicular bisectors.
        # In 3D, the circumcenter of a spherical triangle is the unit vector in the direction of
        # the cross product of (p1 x p2) x (p2 x p3)? No.
        
        # Simpler: the circumcenter c satisfies c · p1 = c · p2 = c · p3 = cos(r).
        # So c is in the null space of [p1 - p2; p1 - p3] ... no, that's for Euclidean.
        # For spherical: c · p1 = c · p2 => c · (p1 - p2) = 0
        # c · p1 = c · p3 => c · (p1 - p3) = 0
        # So c is orthogonal to (p1 - p2) and (p1 - p3).
        # Thus c is parallel to (p1 - p2) x (p1 - p3).
        
        v1 = facet_pts[0] - facet_pts[1]
        v2 = facet_pts[0] - facet_pts[2]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate case, use centroid
            c = np.sum(facet_pts, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c = c / c_norm
        else:
            c = c / c_norm
        
        # Check if c is on the correct side (the side away from the origin, i.e., the facet's normal)
        # The facet normal should point outward. The circumcenter should be in the direction of the facet normal.
        # Let's check the dot product with the facet normal.
        normal = np.cross(facet_pts[1] - facet_pts[0], facet_pts[2] - facet_pts[0])
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-12:
            normal = normal / normal_norm
            # The circumcenter should be on the same side as the normal
            if np.dot(c, normal) < 0:
                c = -c
        
        # Now compute the angular distance from c to the vertices
        dots = np.dot(facet_pts, c)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        r_facet = np.max(angles)
        
        # But we need to ensure that c is actually inside the spherical triangle.
        # If not, the worst point is one of the vertices (but that's 0) or on the edge.
        # For a convex hull, the circumcenter should be inside if the triangle is acute.
        # For obtuse triangles, the circumcenter is outside, and the worst point is the midpoint of the longest edge?
        # Actually, for the covering problem, if the circumcenter is outside the triangle,
        # the farthest point in the triangle from the vertices is the midpoint of the longest edge.
        
        # Let's check if c is inside the spherical triangle.
        # A point c is inside the spherical triangle (p1, p2, p3) if and only if
        # c can be expressed as a positive combination of p1, p2, p3 (with sum of coefficients = 1? no, for spherical it's different).
        # Simpler check: the barycentric coordinates.
        # Or check that c is on the same side of each edge as the opposite vertex.
        
        # Edge p1-p2: the normal is (p1 x p2). c should be on the same side as p3.
        n12 = np.cross(facet_pts[0], facet_pts[1])
        if np.dot(n12, c) * np.dot(n12, facet_pts[2]) < 0:
            # c is outside
            # Find the worst point on the edge p1-p2
            # The worst point is the midpoint of the longest edge
            d12 = np.arccos(np.clip(np.dot(facet_pts[0], facet_pts[1]), -1, 1))
            d23 = np.arccos(np.clip(np.dot(facet_pts[1], facet_pts[2]), -1, 1))
            d31 = np.arccos(np.clip(np.dot(facet_pts[2], facet_pts[0]), -1, 1))
            if d12 >= d23 and d12 >= d31:
                # Midpoint of p1-p2
                mid = facet_pts[0] + facet_pts[1]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[0]), -1, 1))
            elif d23 >= d12 and d23 >= d31:
                mid = facet_pts[1] + facet_pts[2]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[1]), -1, 1))
            else:
                mid = facet_pts[2] + facet_pts[0]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[2]), -1, 1))
        
        # Check other edges similarly
        n23 = np.cross(facet_pts[1], facet_pts[2])
        if np.dot(n23, c) * np.dot(n23, facet_pts[0]) < 0:
            d12 = np.arccos(np.clip(np.dot(facet_pts[0], facet_pts[1]), -1, 1))
            d23 = np.arccos(np.clip(np.dot(facet_pts[1], facet_pts[2]), -1, 1))
            d31 = np.arccos(np.clip(np.dot(facet_pts[2], facet_pts[0]), -1, 1))
            if d12 >= d23 and d12 >= d31:
                mid = facet_pts[0] + facet_pts[1]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[0]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
            elif d23 >= d12 and d23 >= d31:
                mid = facet_pts[1] + facet_pts[2]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[1]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
            else:
                mid = facet_pts[2] + facet_pts[0]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[2]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
        
        n31 = np.cross(facet_pts[2], facet_pts[0])
        if np.dot(n31, c) * np.dot(n31, facet_pts[1]) < 0:
            d12 = np.arccos(np.clip(np.dot(facet_pts[0], facet_pts[1]), -1, 1))
            d23 = np.arccos(np.clip(np.dot(facet_pts[1], facet_pts[2]), -1, 1))
            d31 = np.arccos(np.clip(np.dot(facet_pts[2], facet_pts[0]), -1, 1))
            if d12 >= d23 and d12 >= d31:
                mid = facet_pts[0] + facet_pts[1]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[0]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
            elif d23 >= d12 and d23 >= d31:
                mid = facet_pts[1] + facet_pts[2]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[1]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
            else:
                mid = facet_pts[2] + facet_pts[0]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_candidate = np.arccos(np.clip(np.dot(mid, facet_pts[2]), -1, 1))
                    if r_candidate > r_facet:
                        r_facet = r_candidate
        
        if r_facet > max_radius:
            max_radius = r_facet
    
    return max_radius * 180.0 / np.pi

def _find_worst_point(pts):
    """Find the point on the sphere that is farthest from the nearest site."""
    pts = _normalize(pts)
    # Use the convex hull approach
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid search
        n_grid = 300
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2*np.pi, 2*n_grid, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        dots = grid_pts @ pts.T
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        idx = np.argmax(angles)
        return grid_pts[idx]
    
    max_radius = 0.0
    worst_point = None
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        v1 = facet_pts[0] - facet_pts[1]
        v2 = facet_pts[0] - facet_pts[2]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            c = np.sum(facet_pts, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c = c / c_norm
        else:
            c = c / c_norm
        
        normal = np.cross(facet_pts[1] - facet_pts[0], facet_pts[2] - facet_pts[0])
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-12:
            normal = normal / normal_norm
            if np.dot(c, normal) < 0:
                c = -c
        
        dots = np.dot(facet_pts, c)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        r_facet = np.max(angles)
        
        # Check if c is inside the triangle
        n12 = np.cross(facet_pts[0], facet_pts[1])
        n23 = np.cross(facet_pts[1], facet_pts[2])
        n31 = np.cross(facet_pts[2], facet_pts[0])
        
        inside = True
        if np.dot(n12, c) * np.dot(n12, facet_pts[2]) < 0:
            inside = False
        if np.dot(n23, c) * np.dot(n23, facet_pts[0]) < 0:
            inside = False
        if np.dot(n31, c) * np.dot(n31, facet_pts[1]) < 0:
            inside = False
        
        if not inside:
            # Use midpoint of longest edge
            d12 = np.arccos(np.clip(np.dot(facet_pts[0], facet_pts[1]), -1, 1))
            d23 = np.arccos(np.clip(np.dot(facet_pts[1], facet_pts[2]), -1, 1))
            d31 = np.arccos(np.clip(np.dot(facet_pts[2], facet_pts[0]), -1, 1))
            if d12 >= d23 and d12 >= d31:
                mid = facet_pts[0] + facet_pts[1]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[0]), -1, 1))
                    if r_facet > max_radius:
                        max_radius = r_facet
                        worst_point = mid
            elif d23 >= d12 and d23 >= d31:
                mid = facet_pts[1] + facet_pts[2]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[1]), -1, 1))
                    if r_facet > max_radius:
                        max_radius = r_facet
                        worst_point = mid
            else:
                mid = facet_pts[2] + facet_pts[0]
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    mid = mid / mid_norm
                    r_facet = np.arccos(np.clip(np.dot(mid, facet_pts[2]), -1, 1))
                    if r_facet > max_radius:
                        max_radius = r_facet
                        worst_point = mid
        else:
            if r_facet > max_radius:
                max_radius = r_facet
                worst_point = c
    
    if worst_point is None:
        # Fallback
        worst_point = np.array([1.0, 0.0, 0.0])
    
    return worst_point

def _objective_and_grad(x_flat, grid_pts, t_init):
    """Objective: maximize t subject to angular distance <= t for all grid points.
    We minimize -t + penalty for violations."""
    pts = x_flat.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute min angular distance for each grid point
    dots = grid_pts @ pts.T  # (n_grid, N)
    min_dots = np.min(dots, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    
    # The covering radius is the max of angles
    # We want to minimize this
    return np.max(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Create a dense grid for optimization
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Use gradient descent on the smooth approximation
    # Objective: max over grid points of min over sites of arccos(dot(y, p))
    # This is nonsmooth, so we use a soft-max approximation
    
    lr = 0.01
    n_iters = 500
    soft_beta = 50.0  # Temperature for soft-max
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        pts_flat = pts.ravel()
        
        # Compute distances
        dots = grid_pts @ pts.T  # (n_grid, N)
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        
        # Soft-max for smoothness
        max_angle = np.max(angles)
        exp_angles = np.exp(soft_beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Gradient of the soft-max objective w.r.t. pts
        # d/dp arccos(dot(y, p)) = -y / sqrt(1 - dot^2)
        # But we need the gradient of the weighted sum
        
        # For each grid point y_j with weight w_j, the nearest site is the one that gave min_dots[j]
        # Actually, we need to track which site is nearest for each grid point
        min_idx = np.argmin(dots, axis=1)
        
        grad = np.zeros_like(pts)
        for j in range(len(grid_pts)):
            if weights[j] < 1e-10:
                continue
            y = grid_pts[j]
            p = pts[min_idx[j]]
            dot = np.dot(y, p)
            dot = np.clip(dot, -1.0, 1.0)
            denom = np.sqrt(max(1.0 - dot*dot, 1e-12))
            # d arccos(dot) / d p = -y / denom
            grad[min_idx[j]] += weights[j] * (-y / denom)
        
        # Update
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Evaluate
        current_radius = _covering_radius_deg(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.8
    
    # Final polish with SLSQP
    if time.time() < deadline - 30:
        # Use the best points as initial guess
        x0 = best_pts.ravel()
        
        def obj(x):
            pts_temp = _normalize(x.reshape(N, 3))
            return _covering_radius_deg(pts_temp)
        
        try:
            res = minimize(obj, x0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-8})
            if res.success and res.fun < best_radius:
                best_pts = _normalize(res.x.reshape(N, 3))
                best_radius = res.fun
        except Exception:
            pass
    
    # Final check
    if time.time() < deadline:
        final_radius = _covering_radius_deg(best_pts)
        if final_radius < best_radius:
            best_radius = final_radius
    
    return best_pts
# EVOLVE-BLOCK-END
