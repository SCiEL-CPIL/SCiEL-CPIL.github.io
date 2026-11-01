import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_points = None
    best_score = 0.0

    def get_score(points):
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms > 1.0 + 1e-9):
            return 0.0
            
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        min_boundary = np.min(1.0 - norms)
        
        return min(min_pair_dist, min_boundary)

    def soft_min_objective(flat_points, beta):
        points = flat_points.reshape(-1, 2)
        
        # Penalty for being outside
        norms = np.linalg.norm(points, axis=1)
        outside = np.sum(np.maximum(0, norms - 1.0))
        if outside > 1e-6:
            return 1e6 + outside * 1000
            
        # Values to maximize the min of
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        boundary_dists = 1.0 - norms
        
        x_all = np.concatenate([pair_dists, boundary_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # We minimize this, so we return the negative?
        # No, minimize function minimizes.
        # We want to maximize soft_min.
        # So we minimize -soft_min.
        # -soft_min = 1/beta * log(sum(exp(-beta * x)))
        
        y = -beta * x_all
        max_y = np.max(y)
        # If max_y is very negative, exp might underflow to 0.
        # But y = -beta * x. x is positive (distances). So y is negative.
        # If x is large, y is large negative.
        # exp(y) is small.
        # This is numerically stable if we shift by max_y.
        
        exp_y = np.exp(y - max_y)
        sum_exp = np.sum(exp_y)
        if sum_exp == 0:
            return 0.0
        log_sum_exp = max_y + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        return obj

    def gradient(flat_points, beta):
        points = flat_points.reshape(-1, 2)
        norms = np.linalg.norm(points, axis=1)
        
        # If outside, gradient of penalty
        outside_mask = norms > 1.0
        if np.any(outside_mask):
             grad = np.zeros_like(points)
             safe_norms = np.where(norms > 1e-9, norms, 1.0)
             grad[outside_mask] = points[outside_mask] / safe_norms[outside_mask, np.newaxis]
             return (grad * 1000).flatten()

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        boundary_dists = 1.0 - norms
        
        x_all = np.concatenate([pair_dists, boundary_dists])
        y = -beta * x_all
        max_y = np.max(y)
        exp_y = np.exp(y - max_y)
        sum_exp = np.sum(exp_y)
        if sum_exp == 0:
            return np.zeros_like(flat_points)
        probs = exp_y / sum_exp
        
        grad_points = np.zeros_like(points)
        
        # Boundary part
        boundary_probs = probs[-N:]
        safe_norms = np.where(norms > 1e-9, norms, 1.0)
        boundary_grad = boundary_probs[:, np.newaxis] * points / safe_norms[:, np.newaxis]
        grad_points += boundary_grad
        
        # Pairwise part
        pair_probs = probs[:len(pair_dists)]
        W = np.zeros((N, N))
        W[i_idx, j_idx] = pair_probs
        W[j_idx, i_idx] = pair_probs
        
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, None]
        
        V = np.tensordot(W, unit_vecs, axes=([1], [0]))
        grad_points -= 0.5 * V
        
        return grad_points.flatten()

    # Generate starts
    starts = []
    
    # 1. Hexagonal lattice
    spacing = 0.2
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing
            y = j * spacing * np.sqrt(3) + (0.5 * spacing if i % 2 else 0)
            if x**2 + y**2 <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    radii = np.linalg.norm(pts, axis=1)
    idx = np.argsort(radii)
    pts = pts[idx[:N]]
    pts += np.random.normal(0, 0.01, pts.shape)
    starts.append(pts)
    
    # 2. Random uniform in disk
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts)
        
    # 3. Perturbed Lattice
    for _ in range(5):
        base = starts[0].copy()
        base += np.random.normal(0, 0.1, base.shape)
        norms = np.linalg.norm(base, axis=1)
        mask = norms > 1.0
        base[mask] /= norms[mask, np.newaxis] * 1.01
        starts.append(base)

    for init_pts in starts:
        if time.time() > deadline - 5.0: break
        
        current_pts = init_pts.copy()
        beta = 10.0
        
        # Optimize with increasing beta
        while beta < 1000.0 and time.time() < deadline - 5.0:
            flat = current_pts.flatten()
            
            res = minimize(
                soft_min_objective, 
                flat, 
                args=(beta,),
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(-1.0, 1.0)] * (2*N),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success:
                current_pts = res.x.reshape(-1, 2)
            else:
                current_pts = res.x.reshape(-1, 2)
                
            # Check score
            score = get_score(current_pts)
            if score > best_score:
                best_score = score
                best_points = current_pts.copy()
                
            # Increase beta
            beta *= 2.0
            
    if best_points is None:
        # Fallback to initial lattice
        best_points = starts[0]
        
    return best_points
