"""Program for covering the sphere with 130 equal caps using grid-based L-BFGS-B and facet local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 100)
            lon = np.linspace(0, 2*np.pi, 200, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def grid_objective_and_grad(x_flat):
        """
        Compute the maximum angular distance from a dense grid of points on the sphere
        to the nearest of the N configuration points, and its gradient.
        
        x_flat: flat array of shape (3*N,)
        """
        pts = x_flat.reshape(N, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Create a dense grid of points on the sphere
        # Using 50x100 grid for balance between accuracy and speed
        lat = np.linspace(-np.pi/2, np.pi/2, 50)
        lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Compute angular distances from grid points to configuration points
        # dot = grid_pts @ pts.T  (shape: (num_grid, N))
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (num_grid, N)
        
        # For each grid point, find the nearest configuration point
        min_dists = np.min(ang_dists, axis=1)  # (num_grid,)
        min_indices = np.argmin(ang_dists, axis=1)  # (num_grid,)
        
        # The objective is the maximum of min_dists
        max_dist = np.max(min_dists)
        
        # Compute gradient
        # The gradient of the max is the gradient of the term(s) achieving the max.
        # For simplicity, we use a soft-max approximation to get a smooth gradient.
        # Temperature for soft-max
        T = 20.0
        shifted = min_dists - max_dist
        exp_shifted = np.exp(shifted * T)
        weights = exp_shifted / np.sum(exp_shifted)
        
        # Gradient of min_dist for grid point k w.r.t. pts[i]:
        # If pts[i] is the nearest, d/dp_i arccos(p_k · p_i) = -p_k / sqrt(1 - (p_k·p_i)^2)
        # But we need to be careful: the min is a piecewise function.
        # We approximate the gradient of the soft-max of min_dists.
        
        grad_pts = np.zeros_like(pts)
        
        # For each grid point, add its contribution to the gradient
        for k in range(len(grid_pts)):
            w = weights[k]
            if w < 1e-10:
                continue
            
            i = min_indices[k]
            p_k = grid_pts[k]
            p_i = pts[i]
            
            # d/dp_i arccos(p_k · p_i) = -p_k / sqrt(1 - (p_k·p_i)^2)
            dot_ki = np.dot(p_k, p_i)
            dot_ki = np.clip(dot_ki, -1.0, 1.0)
            denom = np.sqrt(max(1e-10, 1.0 - dot_ki * dot_ki))
            
            # The gradient of the distance w.r.t. p_i is -p_k / denom
            # But we want to minimize the distance, so we move p_i in the direction that reduces it.
            # The gradient of the objective (max distance) w.r.t. p_i is w * (-p_k / denom)
            # So we should move p_i in the direction of -grad = w * p_k / denom
            
            grad_pts[i] += w * p_k / denom
        
        # Project gradient onto tangent plane at each point
        for i in range(N):
            p_i = pts[i]
            grad_pts[i] -= np.dot(grad_pts[i], p_i) * p_i
        
        # Flatten gradient
        grad_flat = grad_pts.ravel()
        
        return max_dist, grad_flat

    def facet_local_search(pts, max_iter=100):
        """
        Local search on convex hull facets to reduce the exact covering radius.
        For each facet, compute its circumradius. If it's the largest, try to move
        the vertices to reduce it.
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norms = np.linalg.norm(c_dir, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                    cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                    radii[valid] = r
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                
                max_rad = np.max(radii)
                max_face_idx = np.argmax(radii)
                
                if max_rad < 0.197:  # ~11.3 degrees in radians
                    break
                
                # Try to improve the worst facet
                fa = faces[max_face_idx]
                i, j, k = fa[0], fa[1], fa[2]
                
                # Compute circumcenter of the worst facet
                a_f = pts[i]
                b_f = pts[j]
                c_f = pts[k]
                ab_f = np.cross(a_f, b_f)
                bc_f = np.cross(b_f, c_f)
                ca_f = np.cross(c_f, a_f)
                c_dir_f = ab_f + bc_f + ca_f
                norm_c_f = np.linalg.norm(c_dir_f)
                
                if norm_c_f < 1e-10:
                    continue
                
                circumcenter = c_dir_f / norm_c_f
                
                # Move each vertex slightly toward the circumcenter
                step = 0.01
                improved = False
                
                for idx in [i, j, k]:
                    p = pts[idx]
                    # Direction toward circumcenter, projected to tangent plane
                    direction = circumcenter - np.dot(circumcenter, p) * p
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    new_p = p + step * direction
                    new_p /= np.linalg.norm(new_p)
                    
                    pts[idx] = new_p
                    new_rad = compute_covering_radius_exact(pts)
                    
                    if new_rad < max_rad - 1e-8:
                        improved = True
                        max_rad = new_rad
                    else:
                        pts[idx] = p
                
                if not improved:
                    break
                    
            except Exception:
                break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with L-BFGS-B optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Use best found so far with perturbation
            if best_points is not None:
                rng = np.random.RandomState(300 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Optimize with L-BFGS-B on grid-based objective
        x0 = points.ravel()
        
        result = minimize(
            grid_objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 500}
        )
        
        points = result.x.reshape(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Phase 2: Facet local search
        points = facet_local_search(points, max_iter=50)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 3: Final polish with L-BFGS-B on the best solution
    if best_points is not None and time.time() < deadline - 10.0:
        x0 = best_points.ravel()
        result = minimize(
            grid_objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300}
        )
        
        points = result.x.reshape(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        points = facet_local_search(points, max_iter=30)
        
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
