"""
Program for covering the sphere with 130 equal caps using repulsion-based optimization.
Refined to use a high-resolution soft-min/soft-max objective for accurate gradient descent.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective.
# A higher resolution is needed to accurately identify the covering radius bottleneck.
# 10,000 points on the sphere provide sufficient density for N=130.
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(3000)
GRID_FINE = _make_fib_grid(10000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

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
            
            # Circumcenter of triangle (a,b,c) on unit sphere
            # Normal to the plane: n = (b-a) x (c-a)
            # Circumcenter C is in the plane a + s(b-a) + t(c-a)
            # For unit sphere, we can use the fact that the circumcenter is proportional to a+b+c
            # IF the triangle is acute. If obtuse, the circumcenter is outside the triangle,
            # but the max distance to the nearest vertex is still determined by the longest side's midpoint or the obtuse vertex.
            # Actually, the circumradius of a spherical triangle is the max angular distance from the circumcenter to the vertices.
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                # The circumcenter on the unit sphere is c_dir_normalized
                # The distance to vertex a is arccos(dot(c_dir, a))
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                # If the triangle is obtuse, the circumcenter is on the other side, 
                # but arccos will give > pi/2. The covering radius is the min(r, pi-r)?
                # No, the circumradius is the radius of the smallest circle covering the vertices.
                # If the circumcenter is on the "wrong" side, the circle goes the other way.
                # The angular distance is min(r, pi - r).
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                # Degenerate triangle (collinear on sphere)
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                # Max distance in a degenerate triangle is half the longest side
                max_side = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                radii[~valid] = max_side / 2.0
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def soft_min_max_objective_grad(x_flat, grid_pts, T_min=200.0, T_max=50.0):
        """
        Computes the value and gradient of the soft-max of soft-min distances.
        Objective: max_y min_x ||x - y||
        Approximation:
        1. For each grid point y_j, compute soft-min distance d_j = -1/T_min log(sum_k exp(-T_min dist(y_j, x_k)))
        2. Compute soft-max of d_j: val = 1/T_max log(sum_j exp(T_max d_j))
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        # dist = arccos(dot)
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        
        # We can work in log-space for stability if needed, but exp(-T*d) is standard.
        # To avoid arccos which has singular gradient at 0, we can use a differentiable approximation?
        # Or just use arccos, it's fine unless dot is exactly 1.
        ang_dists = np.arccos(dot)  # M x N

        # 1. Soft-min over points (axis 1)
        # d_j = -1/T_min * log( sum_k exp(-T_min * d_jk) )
        exp_neg_T_min_d = np.exp(-T_min * ang_dists) # M x N
        sum_exp_neg = np.sum(exp_neg_T_min_d, axis=1) # M
        soft_min_dists = -1.0 / T_min * np.log(sum_exp_neg) # M
        
        # Weights for soft-min: w_jk = exp(-T_min * d_jk) / sum_k exp(-T_min * d_jk)
        w_jk = exp_neg_T_min_d / sum_exp_neg[:, np.newaxis] # M x N

        # 2. Soft-max over grid points (axis 0)
        # val = 1/T_max * log( sum_j exp(T_max * d_j) )
        # For numerical stability, subtract max d_j
        max_d = np.max(soft_min_dists)
        exp_T_max_d = np.exp(T_max * (soft_min_dists - max_d)) # M
        sum_exp_pos = np.sum(exp_T_max_d)
        val = max_d + 1.0 / T_max * np.log(sum_exp_pos)
        
        # Weights for soft-max: v_j = exp(T_max * d_j) / sum_j exp(T_max * d_j)
        v_j = exp_T_max_d / sum_exp_pos # M

        # Gradient
        # d(val)/d(pts_norm[k]) = sum_j v_j * (d(d_j)/d(pts_norm[k]))
        # d(d_j)/d(pts_norm[k]) = - sum_k' w_jk' * (d(dist_jk')/d(pts_norm[k']))
        # d(dist_jk)/d(pts_norm[k]) = - (grid_pts[j] - pts_norm[k].dot(grid_pts[j]) * pts_norm[k]) / dist_jk
        # Wait, dist = arccos(dot). d(dist)/d(dot) = -1/sqrt(1-dot^2).
        # d(dot)/d(pts_norm[k]) = grid_pts[j]
        # So d(dist)/d(pts_norm[k]) = - grid_pts[j] / sqrt(1 - dot_jk^2)
        
        denom = np.sqrt(np.maximum(1.0 - dot**2, 1e-12)) # M x N
        # grad_dists[j,k] = -grid_pts[j] / denom[j,k]
        
        # grad_pts_norm[k] = - sum_j v_j * ( - sum_k' w_jk' * grad_dists[j,k'] )
        # = sum_j v_j * sum_k' w_jk' * (grid_pts[j] / denom[j,k'])
        
        # Vectorized:
        # factor[j,k] = v_j * w_jk / denom[j,k]
        # grad_pts_norm[k] = sum_j factor[j,k] * grid_pts[j]
        
        factor = (v_j[:, np.newaxis] * w_jk) / denom # M x N
        grad_pts_norm = factor.T @ grid_pts # N x 3
        
        # Chain rule for normalization
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=500, T_min=200.0, T_max=50.0):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = soft_min_max_objective_grad(x, grid_pts, T_min, T_max)
            return val
        
        def gradient(x):
            _, g = soft_min_max_objective_grad(x, grid_pts, T_min, T_max)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    def facet_polish(points, max_iter=50):
        """
        Iteratively finds the largest facet and moves the opposite point towards the circumcenter 
        or adjusts the vertices to reduce the circumradius.
        Simple heuristic: If a facet has large circumradius, the points are too far apart.
        We can try to move the points towards the centroid of the facet or the circumcenter.
        """
        pts = points.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                # Calculate circumradius for all faces
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norms = np.linalg.norm(c_dir, axis=1)
                
                radii = np.full(len(faces), np.pi) # Default to large
                valid = norms > 1e-10
                if np.any(valid):
                    c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                    cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                    radii[valid] = r
                else:
                    # Fallback
                    d_ab = np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))
                    radii = d_ab
                    
                max_idx = np.argmax(radii)
                if radii[max_idx] < 0.1: # 0.1 rad ~ 5.7 deg, quite good
                    break
                    
                # Get the face and its circumcenter
                f = faces[max_idx]
                p1, p2, p3 = pts[f[0]], pts[f[1]], pts[f[2]]
                
                # Move points slightly towards each other?
                # Or move the circumcenter direction?
                # A better move: project the points onto a slightly smaller sphere? No, they must be on unit sphere.
                # Move the points towards the centroid of the face (which is the circumcenter for equilateral, close for others).
                centroid = (p1 + p2 + p3) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                # Move each vertex towards the centroid by a small amount
                step = 0.01
                new_p1 = (p1 + step * centroid)
                new_p2 = (p2 + step * centroid)
                new_p3 = (p3 + step * centroid)
                
                new_p1 /= np.linalg.norm(new_p1)
                new_p2 /= np.linalg.norm(new_p2)
                new_p3 /= np.linalg.norm(new_p3)
                
                old_pts = pts.copy()
                pts[f[0]] = new_p1
                pts[f[1]] = new_p2
                pts[f[2]] = new_p3
                
                # Check if it improved the global covering radius
                # Using fast approximation
                dot = np.dot(GRID_COARSE, pts.T)
                dot = np.clip(dot, -1.0, 1.0)
                ang_dists = np.arccos(dot)
                min_dists = np.min(ang_dists, axis=1)
                rad_new = np.max(min_dists)
                
                dot_old = np.dot(GRID_COARSE, old_pts.T)
                dot_old = np.clip(dot_old, -1.0, 1.0)
                ang_dists_old = np.arccos(dot_old)
                min_dists_old = np.min(ang_dists_old, axis=1)
                rad_old = np.max(min_dists_old)
                
                if rad_new < rad_old:
                    pass # Keep change
                else:
                    pts = old_pts # Revert
            except Exception:
                break
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Initialize with Fibonacci spiral
    points = fibonacci_spiral(N).astype(np.float64)
    
    # Quick repulsion pass to spread out
    # Simple gradient descent on sum 1/dist^4
    for _ in range(200):
        if time.time() > deadline - 60.0:
            break
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w_grad = 4.0 * (dist_sq ** -2.5)
        np.fill_diagonal(w_grad, 0.0)
        grad = w_grad @ points
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        alpha = 0.05
        points = points + alpha * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        alpha *= 0.95

    # Phase 2: L-BFGS-B with Coarse Grid
    points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=200, T_min=100.0, T_max=20.0)
    rad = compute_covering_radius_exact(points)
    best_points = points.copy()
    best_rad = rad

    # Phase 3: L-BFGS-B with Fine Grid (Refinement)
    points, _ = optimize_with_lbfgsb(best_points, GRID_FINE, maxiter=300, T_min=200.0, T_max=50.0)
    rad = compute_covering_radius_exact(points)
    if rad < best_rad:
        best_points = points.copy()
        best_rad = rad

    # Phase 4: Facet Polish
    points = facet_polish(best_points, max_iter=100)
    rad = compute_covering_radius_exact(points)
    if rad < best_rad:
        best_points = points.copy()
        best_rad = rad

    # Phase 5: Final L-BFGS-B Polish
    if time.time() < deadline - 20.0:
        points, _ = optimize_with_lbfgsb(best_points, GRID_FINE, maxiter=100, T_min=500.0, T_max=100.0)
        rad = compute_covering_radius_exact(points)
        if rad < best_rad:
            best_points = points.copy()
            best_rad = rad

    return best_points

# EVOLVE-BLOCK-END
