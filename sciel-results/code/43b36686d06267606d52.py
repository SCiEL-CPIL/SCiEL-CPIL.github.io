import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def compute_radius(pts):
        # Compute min pairwise distance / 2 and min boundary distance
        # Vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - r_pts)
        return min(min_pair, min_boundary)
    
    def hex_init(n, seed=42):
        rng = np.random.default_rng(seed)
        best_pts = None
        best_r = 0
        
        for trial in range(50):
            # Vary the spacing
            spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 - trial * 0.012 + rng.random() * 0.01)
            if spacing <= 0.001:
                continue
            
            k = int(1.0 / spacing) + 3
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            
            if len(points) < n:
                continue
            
            points = np.array(points)
            # Add small random jitter
            points += rng.random(size=points.shape) * spacing * 0.01
            
            # Select n points closest to center
            dists_center = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists_center)[:n]
            pts_sel = points[idx].copy()
            
            r = compute_radius(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel.copy()
        
        return best_pts, best_r
    
    # Step 1: Initialize with hexagonal lattice
    pts, best_radius = hex_init(N, seed=42)
    best_pts = pts.copy()
    
    # Step 2: Gradient-based optimization with soft-min objective
    # We maximize the minimum of all pairwise distances/2 and boundary distances
    # Using a smooth approximation via log-sum-exp
    
    def soft_min_and_grad(pts, T):
        """Compute soft_min value and gradient for the packing objective."""
        N_local = len(pts)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        dists_half = dists / 2.0
        np.fill_diagonal(dists_half, np.inf)
        
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - r_pts
        
        # All constraint values
        # We need to compute soft_min of all these values
        # For numerical stability, find the min first
        min_pair = np.min(dists_half)
        min_bound = np.min(boundary_dists)
        min_val = min(min_pair, min_bound)
        
        # Compute weights using exp(-(x_k - min_val)/T)
        # For pairwise
        exp_pairs = np.exp(-(dists_half - min_val) / T)
        exp_pairs[np.isinf(dists_half)] = 0.0
        
        # For boundary
        exp_bound = np.exp(-(boundary_dists - min_val) / T)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bound)
        if sum_exp < 1e-30:
            # Degenerate case
            return min_val, np.zeros_like(pts)
        
        w_pairs = exp_pairs / sum_exp
        w_bound = exp_bound / sum_exp
        
        # Gradient computation
        # d(soft_min)/d(pts_i) = sum_j w_pairs[i,j] * (pts_i - pts_j) / dists[i,j] 
        #                        + w_bound[i] * (-pts_i / r_pts[i])
        
        # For pairwise part:
        # (pts_i - pts_j) / dists[i,j] = (pts_i - pts_j) / (2 * dists_half[i,j])
        factor = np.zeros_like(dists_half)
        mask_nonzero = dists_half > 1e-10
        factor[mask_nonzero] = w_pairs[mask_nonzero] / (2.0 * dists_half[mask_nonzero])
        
        sum_factor = np.sum(factor, axis=1)
        force_pairs = pts * sum_factor[:, np.newaxis] - (factor @ pts)
        
        # For boundary part:
        r_safe = np.where(r_pts < 1e-10, 1.0, r_pts)
        force_bound = -w_bound[:, np.newaxis] * pts / r_safe[:, np.newaxis]
        
        grad = force_pairs + force_bound
        
        # Soft min value
        soft_min_val = min_val - T * np.log(sum_exp)
        
        return soft_min_val, grad
    
    # Optimization loop with adaptive step size
    total_budget = 140.0  # seconds for optimization
    T_init = 0.008
    T_min = 0.0001
    
    # Phase 1: Larger temperature, explore
    for phase in range(3):
        if time.time() > deadline:
            break
        
        if phase == 0:
            T_start, T_end = T_init, 0.002
            n_iters = 2000
        elif phase == 1:
            T_start, T_end = 0.002, 0.0005
            n_iters = 2000
        else:
            T_start, T_end = 0.0005, T_min
            n_iters = 2000
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            progress = it / n_iters
            T = T_start * (T_end / T_start) ** progress
            
            soft_val, grad = soft_min_and_grad(pts, T)
            
            # Check actual radius
            actual_r = compute_radius(pts)
            if actual_r > best_radius:
                best_radius = actual_r
                best_pts = pts.copy()
            
            # Adaptive step size
            # Use the norm of the gradient to scale the step
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norm = np.where(grad_norm < 1e-10, 1.0, grad_norm)
            
            # Step size decreases over time
            base_step = 0.05 * (1.0 - progress * 0.7) + 0.005
            step = base_step / (1.0 + np.log1p(np.mean(grad_norm) / T))
            
            # Take step
            pts_new = pts + step * grad / grad_norm
            
            # Project back inside the unit disk
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new > 0.9995
            if np.any(mask_out):
                pts_new[mask_out] = pts_new[mask_out] / r_new[mask_out, np.newaxis] * 0.9995
            
            # Accept if it improves or with some probability
            new_r = compute_radius(pts_new)
            if new_r >= actual_r - 1e-8:
                pts = pts_new
            else:
                # Simulated annealing style acceptance
                delta = new_r - actual_r
                if delta < 0:
                    accept_prob = np.exp(delta / (T * 0.5))
                    if np.random.random() > accept_prob:
                        pts = pts_new
                else:
                    pts = pts_new
    
    # Phase 4: Fine-tuning with very small steps
    if time.time() < deadline - 2:
        T = 0.0001
        for it in range(3000):
            if time.time() > deadline - 1:
                break
            
            soft_val, grad = soft_min_and_grad(pts, T)
            
            actual_r = compute_radius(pts)
            if actual_r > best_radius:
                best_radius = actual_r
                best_pts = pts.copy()
            
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norm = np.where(grad_norm < 1e-10, 1.0, grad_norm)
            
            step = 0.002
            pts_new = pts + step * grad / grad_norm
            
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new > 0.9998
            if np.any(mask_out):
                pts_new[mask_out] = pts_new[mask_out] / r_new[mask_out, np.newaxis] * 0.9998
            
            new_r = compute_radius(pts_new)
            if new_r >= actual_r - 1e-10:
                pts = pts_new
    
    # Final check
    final_r = compute_radius(best_pts)
    if final_r > best_radius:
        best_radius = final_r
    
    return best_pts
