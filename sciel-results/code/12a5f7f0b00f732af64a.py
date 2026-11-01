import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid(n_grid=50000):
    """Generate a dense spherical grid for the surrogate objective."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _sinkhorn_divergence(pts, grid, eps=0.05, n_iter=20):
    """
    Compute Sinkhorn divergence between uniform measure on grid and 
    measure induced by points (approximated by Voronoi cells or just uniform assignment).
    
    Here we simplify: we treat the problem as minimizing the maximum distance 
    but using a soft assignment gradient.
    
    Objective: sum_i exp(-d_i^2 / eps) where d_i is distance to nearest point.
    Gradient with respect to point p_k: sum_{i assigned to k} (p_k - g_i) / d_i * weight_i
    """
    # Compute distances from grid to all points
    # grid: (M, 3), pts: (N, 3)
    # dist_sq: (M, N)
    
    # To save memory and time, we process in chunks if M is large, but 50k x 130 is ~6.5M floats ~ 50MB. OK.
    # However, computing full distance matrix every step is expensive.
    # Let's use a chunked approach for the gradient calculation.
    
    M = len(grid)
    chunk_size = 10000
    obj_val = 0.0
    grad = np.zeros_like(pts)
    
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        g_chunk = grid[start:end]
        
        # dist_sq: (chunk, N)
        dist_sq = np.sum(g_chunk**2, axis=1, keepdims=True) + np.sum(pts**2, axis=1, keepdims=True).T - 2 * g_chunk @ pts.T
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)
        
        # Soft assignment weights: softmax over negative distances
        # We want to minimize the max, so we emphasize the far points.
        # Let's use a temperature T. Low T = hard max. High T = smooth.
        T = 0.01 
        weights = np.exp(-dist / T)
        weights /= (np.sum(weights, axis=1, keepdims=True) + _EPS)
        
        # Objective contribution: sum of weighted distances (approximating max)
        # Actually, for covering radius, we care about the maximum.
        # A common surrogate is log-sum-exp of distances.
        # grad of log-sum-exp(d_i) wrt p_k is sum_i (d_i / LSE) * (p_k - g_i)/d_i for i where p_k is nearest?
        # No, differentiable max is tricky.
        
        # Let's use the "soft max" objective: log(sum exp(d_i/T))
        # Gradient wrt p_k: sum_i (exp(d_i/T)/sum) * (1/T) * (p_k - g_i)/d_i * (if p_k is the one affecting d_i)
        # This is complex because d_i = min_j ||g_i - p_j||.
        
        # Simpler approach: use the gradient of the sum of squared distances to the NEAREST point,
        # but weighted by how "bad" that distance is.
        
        # Let's stick to a simpler differentiable proxy for the covering radius:
        # Minimize: log( sum_i exp( ||g_i - nearest(g_i)||^2 / T ) )
        
        # Find nearest point for each grid point
        min_dist_sq = np.min(dist_sq, axis=1)
        min_idx = np.argmin(dist_sq, axis=1)
        
        # Soft max objective
        # L = log( sum exp( min_dist_sq / T ) )
        # This is hard to differentiate through argmin.
        
        # Alternative: Use the "entropic regularization" of the optimal transport cost.
        # Cost matrix C_{ij} = ||g_i - p_j||^2
        # Sinkhorn divergence S_eps(mu, nu) where mu is uniform on grid, nu is uniform on pts.
        # This encourages the points to be well-distributed.
        
        # For efficiency, let's just use the gradient of the sum of distances to the nearest point,
        # weighted by an exponential of the distance (to focus on the worst).
        
        # Weight w_i = exp( min_dist_sq_i / T )
        w = np.exp(min_dist_sq / T)
        w_sum = np.sum(w)
        w /= (w_sum + _EPS)
        
        # Gradient of sum_i w_i * min_dist_i
        # min_dist_i = ||g_i - p_{k_i}||
        # d/dp_k ( ||g_i - p_k|| ) = (p_k - g_i) / ||g_i - p_k||
        
        for k in range(N):
            mask = (min_idx == k)
            if not np.any(mask):
                continue
            g_k = g_chunk[mask]
            d_k = np.sqrt(min_dist_sq[mask])
            # Avoid division by zero
            d_k = np.maximum(d_k, _EPS)
            
            # grad_k = sum_{i in k} w_i * (p_k - g_i) / d_i
            # Vectorized:
            terms = (pts[k][None, :] - g_k) / d_k[:, None]
            grad[k] += np.sum(w[mask, None] * terms, axis=0)
            
        obj_val += np.sum(w * np.sqrt(min_dist_sq))
        
    # Normalize gradient
    grad /= M
    
    return obj_val, grad

def _riemannian_grad_descent(pts, grid, n_iter=100, lr=0.05, deadline=None):
    """
    Perform Riemannian gradient descent on the sphere.
    """
    best_pts = pts.copy()
    best_obj = np.inf
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
            
        obj, grad = _sinkhorn_divergence(pts, grid, eps=0.01, n_iter=1)
        
        # Project gradient to tangent space
        # grad_tan = grad - (grad . p) p
        proj = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tan = grad - proj * pts
        
        # Update
        new_pts = pts - lr * grad_tan
        # Re-normalize
        new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check improvement in the ACTUAL metric (hull radius) periodically?
        # Or just trust the surrogate?
        # To be safe, let's check the actual hull radius every few steps.
        if it % 10 == 0:
            actual_r = _hull_radius(new_pts)
            if actual_r < best_obj:
                best_obj = actual_r
                best_pts = new_pts.copy()
        
        pts = new_pts
        
    # Final check
    final_r = _hull_radius(pts)
    if final_r < best_obj:
        best_obj = final_r
        best_pts = pts.copy()
        
    return best_pts, best_obj

def _polish_with_hull(pts, deadline=None):
    """
    Polish using the original hull-based pole chasing for the final few steps.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii = np.arccos(cos_theta)
            
            curr_r = np.max(radii)
            if curr_r < best_r - 1e-12:
                best_r = curr_r
                best_pts = pts.copy()
                
            worst_idx = np.argmax(radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            step = 0.05
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    continue
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            cand_r = _hull_radius(cand)
            if cand_r < curr_r - 1e-14:
                pts = cand
            else:
                break
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    # Initial points
    pts = _initial_fib()
    
    # Generate grid
    grid = _get_grid(50000)
    
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    # Phase 1: Riemannian Gradient Descent on Sinkhorn-like objective
    # We use a moderate number of iterations to avoid getting stuck.
    if time.time() < deadline - 60.0:
        pts, r = _riemannian_grad_descent(pts, grid, n_iter=200, lr=0.02, deadline=deadline - 60.0)
        if r < best_r:
            best_r = r
            best_pts = pts
    
    # Phase 2: Polish with hull-based methods
    if time.time() < deadline - 10.0:
        pts, r = _polish_with_hull(best_pts, deadline=deadline - 5.0)
        if r < best_r:
            best_r = r
            best_pts = pts
    
    # Phase 3: Try a few random restarts with shorter GD runs if time permits
    time_left = deadline - time.time()
    if time_left > 30:
        n_restarts = int(time_left / 20)
        for i in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            rng = np.random.default_rng(i)
            perturbed = best_pts + rng.standard_normal((N, 3)) * 0.1
            perturbed /= (np.linalg.norm(perturbed, axis=1, keepdims=True) + _EPS)
            
            pts, r = _riemannian_grad_descent(perturbed, grid, n_iter=50, lr=0.02, deadline=deadline - 5.0)
            if r < best_r:
                best_r = r
                best_pts = pts
            
            # Quick polish
            if time.time() < deadline - 2.0:
                pts, r = _polish_with_hull(pts, deadline=deadline - 2.0)
                if r < best_r:
                    best_r = r
                    best_pts = pts

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
