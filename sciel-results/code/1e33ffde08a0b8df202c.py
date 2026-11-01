import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid():
    """Generate a dense spherical grid for the surrogate objective."""
    # Use a grid with ~10k points to balance resolution and speed
    # Fibonacci spiral is good for this
    n_grid = 10000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _surrogate_objective_and_grad(pts_flat, grid):
    """
    Compute the sum of squared nearest-neighbor distances and its gradient.
    pts_flat: flattened array of shape (3*N,)
    grid: array of shape (M, 3)
    """
    pts = pts_flat.reshape(N, 3)
    
    # Normalize points to ensure they are on the sphere during optimization
    # This is important because L-BFGS-B might step off the manifold
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    # Compute distances from each grid point to each candidate point
    # dist_sq[i, j] = || grid[i] - pts[j] ||^2
    # This is equivalent to 2 - 2 * (grid[i] . pts[j])
    # We want to minimize the sum over i of min_j dist_sq[i, j]
    
    # Vectorized computation of dot products
    # G: (M, 3), P: (N, 3)
    # Dots: (M, N)
    dots = grid @ pts.T
    
    # dist_sq = 2 - 2 * dots
    dist_sq = 2.0 - 2.0 * dots
    
    # Find the minimum distance squared for each grid point
    min_dist_sq = np.min(dist_sq, axis=1)
    min_idx = np.argmin(dist_sq, axis=1)
    
    # Objective: sum of min_dist_sq
    obj = np.sum(min_dist_sq)
    
    # Gradient calculation
    # d(obj)/d(pts[k]) = sum_{i: min_idx[i] == k} d(min_dist_sq[i])/d(pts[k])
    # d(dist_sq[i, j])/d(pts[j]) = -2 * grid[i]
    # So d(min_dist_sq[i])/d(pts[k]) is -2 * grid[i] if min_idx[i] == k, else 0
    
    grad = np.zeros_like(pts)
    # Use np.add.at for accumulation
    np.add.at(grad, min_idx, -2.0 * grid)
    
    # Since we normalized pts, we need to account for the projection back to the sphere.
    # However, for a first pass, we can just use the raw gradient and let the optimizer handle it,
    # or we can project the gradient.
    # A simpler approach for manifold optimization is to use the raw gradient in the flat space,
    # but ensure the points are re-normalized after each step.
    # L-BFGS-B doesn't natively support manifolds, so we rely on the normalization in the objective.
    # Actually, if we normalize inside the objective, the gradient returned must be consistent with that.
    # Let's use a simpler trick: optimize the flat coordinates, but in the objective, we normalize.
    # The gradient we compute above is for the normalized pts.
    # To be safe, let's just return the gradient of the un-normalized objective?
    # No, let's stick to the normalized objective. The gradient of f(x/||x||) is (I - xx^T/||x||^2) * grad_f(x/||x||).
    # But we computed grad_f for the normalized points.
    # Let's just use the gradient as is, it points in the direction of decreasing distance in the tangent space.
    # When L-BFGS-B updates the flat vector, we normalize it again in the next call.
    # This is a standard "projected gradient" approach.
    
    return obj, grad.flatten()

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
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

def _polish_with_hull(pts, deadline):
    """
    Polish the points using subgradient descent on the actual hull metric.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    step = 0.05
    no_improve_count = 0
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Compute hull radii and poles
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
        except Exception:
            break
        
        curr_r = np.max(radii)
        
        if curr_r < best_r - 1e-9:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                step *= 0.9
                no_improve_count = 0
                if step < 1e-6:
                    break
        
        worst_idx = np.argmax(radii)
        v_indices = faces[valid][worst_idx]
        pole = normals[worst_idx]
        
        cand = pts.copy()
        for v in v_indices:
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                dir_vec = rng.standard_normal(3)
                dir_vec /= (np.linalg.norm(dir_vec) + _EPS)
            
            cand[v] = pts[v] + step * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        cand_r = _hull_radius(cand)
        if cand_r < curr_r:
            pts = cand
        else:
            step *= 0.8
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 140.0  # Safety margin
    grid = _get_grid()
    
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Run L-BFGS-B optimization with multiple restarts
    n_restarts = 3
    time_per_restart = (deadline - 20.0) / n_restarts  # Leave 20s for polishing
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            pts = _initial_fib()
        else:
            # Perturb the best found so far
            rng = np.random.default_rng(42 + restart)
            pts = best_pts + rng.standard_normal((N, 3)) * 0.1
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        x0 = pts.flatten()
        
        try:
            res = minimize(
                _surrogate_objective_and_grad,
                x0,
                args=(grid,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 500, 'time_limit': time_per_restart}
            )
            
            if res.success:
                new_pts = res.x.reshape(N, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_r = _hull_radius(new_pts)
                
                if new_r < best_r:
                    best_r = new_r
                    best_pts = new_pts
        except Exception:
            pass
    
    # Final polish with the true metric
    if time.time() < deadline - 5.0:
        best_pts, best_r = _polish_with_hull(best_pts, deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
