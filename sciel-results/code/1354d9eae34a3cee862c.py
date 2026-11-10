import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius_and_centers(points):
    """
    Computes the covering radius (max circumradius of hull facets) 
    and returns the radius and the circumcenter of the worst facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None

    max_radius = 0.0
    worst_center = None
    worst_facet_idx = -1

    for i, simplex in enumerate(hull.simplices):
        # Get the 3 points of the facet
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]
        
        # Compute Gram matrix
        # M = [[u.u, u.v, u.w],
        #      [v.u, v.v, v.w],
        #      [w.u, w.v, w.w]]
        # Since u,v,w are unit vectors, diagonals are 1.
        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)
        
        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])
        
        # Solve M x = 1
        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            # Degenerate facet, skip or handle
            continue
            
        xT_M_x = np.dot(x, M @ x)
        
        if xT_M_x <= 0:
            continue
            
        cos_R = 1.0 / np.sqrt(xT_M_x)
        
        # Clip to valid range
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        # The circumcenter vector C is proportional to x_1 u + x_2 v + x_3 w
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        # The radius is arccos(cos_R). 
        # Note: cos_R is the dot product of C and u.
        # We should verify sign. For a convex hull facet, the origin is inside, 
        # so the circumcenter should be on the same side as the interior?
        # Actually, the circumcenter of the spherical triangle is the point on the sphere 
        # equidistant to the vertices. 
        # The distance is R = arccos(C . u).
        
        R = math.acos(cos_R)
        
        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

    return math.degrees(max_radius), worst_center, worst_facet_idx, hull

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Initial points: Fibonacci spiral with slight jitter
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random jitter to break symmetries
    noise = np.random.randn(N, 3) * 0.01
    points = points + noise
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_radius_and_centers(best_points)
    
    # Phase 1: Repulsive force to spread points
    # This helps avoid local minima where points are clustered
    # We use a simple gradient descent on sum of 1/d^2
    alpha = 0.01
    for step in range(200):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Force is proportional to 1/dist^3 (gradient of 1/dist^2 is -2/dist^3 * diff)
                # We want to repel, so force on i is + diff / dist^3
                force = diff / (dist_sq ** 1.5)
                forces[i] += force
                forces[j] -= force
        
        # Normalize forces to ensure stability
        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm == 0, 1, forces_norm)
        
        new_points = points + alpha * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
        if radius < best_radius:
            best_radius = radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If it gets worse, reduce alpha and try again or stop phase 1
            alpha *= 0.9
            if alpha < 0.001:
                break
        
    # Phase 2: Local search targeting the largest facet
    # Move vertices of the worst facet towards its circumcenter
    alpha = 0.1
    for step in range(500):
        if time.time() > deadline:
            break
            
        radius, center, facet_idx, hull = _compute_covering_radius_and_centers(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if center is None or facet_idx == -1:
            break
            
        # Get the vertices of the worst facet
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Move these vertices towards the center
        # The direction on the sphere from point p to center c is (c - (c.p)p)
        # We want to move p towards c.
        
        def move_towards(p, c, step_size):
            # Project c onto the tangent plane at p
            # Tangent vector t = c - (c.p)p
            dot_cp = np.dot(c, p)
            t = c - dot_cp * p
            norm_t = np.linalg.norm(t)
            if norm_t < 1e-10:
                return p
            t = t / norm_t
            new_p = p + step_size * t
            new_p = new_p / np.linalg.norm(new_p)
            return new_p
        
        new_points = points.copy()
        
        # Try moving all 3 vertices
        # We can also try moving them individually to find the best move
        # For simplicity, move all 3 slightly
        new_points[v0] = move_towards(points[v0], center, alpha)
        new_points[v1] = move_towards(points[v1], center, alpha)
        new_points[v2] = move_towards(points[v2], center, alpha)
        
        new_radius, _, _, _ = _compute_covering_radius_and_centers(new_points)
        
        if new_radius < radius:
            points = new_points
            # Maybe increase alpha slightly if improving fast?
            # alpha = min(0.2, alpha * 1.05)
        else:
            # If moving all 3 doesn't work, try moving them one by one
            improved = False
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = move_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius_and_centers(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break
            
            if not improved:
                # Reduce step size
                alpha *= 0.9
                if alpha < 0.001:
                    # Try random restarts or perturbations
                    # Add small random perturbation to points
                    noise = np.random.randn(N, 3) * 0.01
                    perturbed = points + noise
                    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                    pert_radius, _, _, _ = _compute_covering_radius_and_centers(perturbed)
                    if pert_radius < best_radius:
                        points = perturbed
                        best_points = points.copy()
                        best_radius = pert_radius
                        alpha = 0.05
                    else:
                        break
                        
    return best_points
