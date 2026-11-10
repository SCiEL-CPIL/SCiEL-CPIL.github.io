import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _covering_radius(points):
    """
    Calculate the exact covering radius in radians.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        d = np.abs(np.dot(normal, p0)) / norm_normal
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _gradient_descent_polish(points, steps=200, beta=20.0, alpha=0.01):
    """
    Gradient descent on the sphere using circumcenter forces.
    """
    points = points.copy()
    
    for step in range(steps):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        # Compute circumradii and circumcenters for all facets
        radii = []
        centers = []
        for simplex in hull.simplices:
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]
            
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_normal = np.linalg.norm(normal)
            
            if norm_normal < 1e-10:
                radii.append(0.0)
                centers.append(np.zeros(3))
                continue
            
            d = np.abs(np.dot(normal, p0)) / norm_normal
            d = np.clip(d, 0.0, 1.0)
            r = np.arccos(d)
            
            # Circumcenter is the normalized normal
            # Ensure it points to the "outside" (away from origin if triangle is small)
            # Actually, for a convex hull of points on the sphere, the circumcenter of the spherical triangle
            # is the point on the sphere furthest from the triangle's interior.
            # The vector normal points to one of the two poles.
            # We need the pole that is the circumcenter.
            # The circumcenter u satisfies u . p0 = u . p1 = u . p2 = cos(r).
            # So u = normal / ||normal|| * sign.
            # Let's check the dot product.
            u = normal / norm_normal
            if np.dot(u, p0) < 0:
                u = -u
            # Verify
            # cos_r = np.dot(u, p0)
            # r_check = np.arccos(cos_r)
            
            radii.append(r)
            centers.append(u)
            
        radii = np.array(radii)
        centers = np.array(centers)
        
        # Compute weights
        max_r = np.max(radii)
        if max_r == 0:
            break
        weights = np.exp(beta * (radii - max_r))
        
        # Accumulate forces
        forces = np.zeros_like(points)
        
        for i, simplex in enumerate(hull.simplices):
            w = weights[i]
            if w < 1e-6:
                continue
                
            u = centers[i]
            r = radii[i]
            
            # Gradient of angular distance from p to u
            # grad_p dist(p, u) = (p - (p.u)u) / sin(r)
            sin_r = np.sin(r)
            if sin_r < 1e-6:
                continue
                
            for idx in simplex:
                p = points[idx]
                # p - (p.u)u is the tangential component
                proj = np.dot(p, u) * u
                grad = (p - proj) / sin_r
                # We want to minimize r, so we move in the direction of -grad
                forces[idx] -= w * grad
                
        # Update points
        points += alpha * forces
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        # Adaptive step size? Keep it simple for now.
        
    return points

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Repulsive Relaxation (Coarse)
    # Simple repulsive force to spread points
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_dist_cubed, 0.0)
        forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        points += 0.01 * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
    current_radius = _covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius
        
    # 3. Gradient Descent Polish
    # Run multiple rounds with decreasing step size
    for round_num in range(5):
        if time.time() > deadline - 5:
            break
            
        # Use a smaller step size in later rounds
        step_size = 0.05 / (round_num + 1)
        steps = 100 if round_num < 3 else 50
        
        polished = _gradient_descent_polish(best_points, steps=steps, beta=20.0, alpha=step_size)
        rad = _covering_radius(polished)
        
        if rad < best_radius:
            best_points = polished
            best_radius = rad
            
    # 4. Final Nelder-Mead if time permits (for local non-smooth improvements)
    if time.time() < deadline - 10:
        def objective(params):
            pts = np.zeros((N, 3))
            for i in range(N):
                theta = params[2*i]
                phi = params[2*i+1]
                pts[i] = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            return _covering_radius(pts)
            
        params0 = np.zeros(2*N)
        for i in range(N):
            p = best_points[i]
            theta = np.arccos(np.clip(p[2], -1, 1))
            phi = np.arctan2(p[1], p[0])
            params0[2*i] = theta
            params0[2*i+1] = phi
            
        try:
            res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5})
            if res.fun < best_radius:
                # Convert back
                final_pts = np.zeros((N, 3))
                for i in range(N):
                    theta = res.x[2*i]
                    phi = res.x[2*i+1]
                    final_pts[i] = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
                best_points = final_pts
                best_radius = res.fun
        except:
            pass

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points /= norms
    
    return best_points
