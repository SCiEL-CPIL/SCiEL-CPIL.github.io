"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Smooth min function for wall distance
    beta_wall = 50.0
    
    def wall_distance(pts):
        """Calculate smooth minimum distance to walls."""
        # pts: (N, 3)
        # dist to each wall: 0.5 - |x|, 0.5 - |y|, 0.5 - |z|
        d = 0.5 - np.abs(pts)
        # Smooth min: -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability, subtract max
        max_d = np.max(d, axis=1, keepdims=True)
        exp_d = np.exp(-beta_wall * (d - max_d))
        log_sum = np.log(np.sum(exp_d, axis=1, keepdims=True))
        smooth_min = -log_sum / beta_wall + max_d
        return smooth_min.flatten()

    def penalty_and_grad(x, r):
        """
        Calculate penalty for violating radius r constraints and its gradient.
        Penalty = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - b_i)^2
        Using smooth hinge: h(u) = 0.5 * (u + sqrt(u^2 + eps))
        Penalty term = h(u)^2
        """
        pts = x.reshape(N, 3)
        eps = 1e-8
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Only consider i < j to avoid double counting
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_ij = dists[i_idx, j_idx]
        
        # u_ij = 2r - d_ij
        u_ij = 2.0 * r - d_ij
        
        # Smooth hinge h(u)
        sqrt_u2 = np.sqrt(u_ij**2 + eps)
        h_ij = 0.5 * (u_ij + sqrt_u2)
        
        # Penalty sum
        pen_pair = np.sum(h_ij**2)
        
        # Gradient of h(u) wrt u
        dh_du = 0.5 * (1.0 + u_ij / sqrt_u2)
        
        # Gradient of penalty wrt u
        dpen_du = 2.0 * h_ij * dh_du
        
        # Gradient wrt pts
        # d(u_ij)/d(pts_i) = - (pts_i - pts_j) / d_ij
        # d(u_ij)/d(pts_j) = (pts_i - pts_j) / d_ij
        diff_ij = diff[i_idx, j_idx] # pts_i - pts_j
        inv_d = 1.0 / np.maximum(d_ij, 1e-8)
        
        grad_pair = np.zeros_like(pts)
        # Sum contributions to pts_i
        # dpen/dpts_i = sum_j dpen_du * (-diff_ij / d_ij)
        # We need to accumulate into grad_pair[i_idx] and grad_pair[j_idx]
        
        # Use np.add.at for accumulation
        vec = dpen_du[:, np.newaxis] * (-diff_ij * inv_d[:, np.newaxis])
        np.add.at(grad_pair, i_idx, vec)
        
        vec_j = dpen_du[:, np.newaxis] * (diff_ij * inv_d[:, np.newaxis])
        np.add.at(grad_pair, j_idx, vec_j)
        
        # Wall distance
        b_i = wall_distance(pts)
        u_wall = r - b_i
        
        # Smooth hinge for wall
        sqrt_uw2 = np.sqrt(u_wall**2 + eps)
        h_wall = 0.5 * (u_wall + sqrt_uw2)
        
        pen_wall = np.sum(h_wall**2)
        
        dh_dw = 0.5 * (1.0 + u_wall / sqrt_uw2)
        dpen_dw = 2.0 * h_wall * dh_dw
        
        # Gradient of b_i wrt pts
        # b_i is smooth min of d_k = 0.5 - |x_k|
        # d(d_k)/d(x_k) = -sign(x_k)
        # d(b_i)/d(x_k) = sum_k exp(-beta(d_k - max)) / sum(exp(...)) * (-sign(x_k))
        # Let's recompute weights for gradient
        d = 0.5 - np.abs(pts)
        max_d = np.max(d, axis=1, keepdims=True)
        exp_d = np.exp(-beta_wall * (d - max_d))
        weights = exp_d / np.sum(exp_d, axis=1, keepdims=True)
        
        signs = np.sign(pts)
        # If pts is 0, sign is 0. We can use 1 or random, but 0 is fine as subgradient.
        # Actually, if pts is 0, derivative of |x| is undefined. 
        # But we are in the interior, so pts != 0 usually.
        
        # grad_b_i_k = weights[i,k] * (-sign(pts[i,k]))
        grad_b = -weights * signs
        
        # dpen/dpts = sum_i dpen_dw_i * grad_b_i
        # dpen_dw is (N,), grad_b is (N, 3)
        grad_wall = dpen_dw[:, np.newaxis] * grad_b
        
        total_pen = pen_pair + pen_wall
        total_grad = grad_pair + grad_wall
        
        return total_pen, total_grad.flatten()

    def calculate_score(centers):
        if centers is None:
            return 0.0
        pts = centers.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # True min wall distance
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def init_grid(perturb=0.01, seed=0):
        rng = np.random.RandomState(seed)
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
        grid = grid + rng.uniform(-perturb, perturb, size=grid.shape)
        return np.clip(grid, -0.45, 0.45)

    def init_random(seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    starts = [
        init_grid(perturb=0.01, seed=0),
        init_grid(perturb=0.02, seed=1),
        init_random(seed=2)
    ]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        x0 = start_pts.flatten()
        
        # Binary search on r
        r_low = 0.10
        r_high = 0.12
        
        best_r_in_this_attempt = r_low
        best_x_in_this_attempt = x0
        
        # Limit number of binary search steps
        max_bs_steps = 12
        for _ in range(max_bs_steps):
            if time.time() > deadline - 5:
                break
                
            r_mid = (r_low + r_high) / 2.0
            
            # Optimize penalty for r_mid
            try:
                res = minimize(penalty_and_grad, x0, method='L-BFGS-B', 
                               args=(r_mid,),
                               bounds=bounds,
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                
                # Check if feasible
                pen_val = res.fun
                if pen_val < 1e-4:
                    # Feasible
                    r_low = r_mid
                    best_r_in_this_attempt = r_mid
                    best_x_in_this_attempt = res.x.copy()
                else:
                    # Infeasible
                    r_high = r_mid
                    
            except Exception:
                r_high = r_mid
                continue
            
            # If range is small, stop
            if r_high - r_low < 1e-5:
                break
            
            # Update x0 for next iteration to warm start
            x0 = res.x if hasattr(res, 'x') and res.x is not None else x0

        # Evaluate the best found in this attempt
        if best_x_in_this_attempt is not None:
            pts = best_x_in_this_attempt.reshape(N, 3)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_centers = pts.copy()
                
    if best_centers is None:
        best_centers = init_grid()
        
    return best_centers
# EVOLVE-BLOCK-END
