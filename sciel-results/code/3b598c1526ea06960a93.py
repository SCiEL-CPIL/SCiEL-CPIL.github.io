"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def score(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def objective(pts_flat, k=50.0):
        """
        Objective function to minimize.
        We use a smooth approximation of (max/min)^2.
        
        Let d_min = min(d_i), d_max = max(d_i).
        We want to minimize (d_max / d_min)^2.
        
        Approximation:
        d_max ≈ logsumexp(k * d_i) / k
        d_min ≈ -logsumexp(-k * d_i) / k
        
        Ratio ≈ exp(log(d_max) - log(d_min))
        Ratio^2 ≈ exp(2 * (log(d_max) - log(d_min)))
        
        This is smooth and differentiable.
        """
        pts_2d = pts_flat.reshape(N, 2)
        diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d = np.maximum(d, 1e-6)
        
        # Compute smooth max and min
        # d_max_smooth = (1/k) * log(sum(exp(k * d)))
        # d_min_smooth = -(1/k) * log(sum(exp(-k * d)))
        
        # To avoid overflow, use logsumexp trick
        # log(sum(exp(k*d))) = k*max_d + log(sum(exp(k*(d - max_d))))
        max_d = np.max(d)
        log_sum_exp_k_d = k * max_d + np.log(np.sum(np.exp(k * (d - max_d))))
        d_max_smooth = log_sum_exp_k_d / k
        
        min_d = np.min(d)
        log_sum_exp_neg_k_d = -k * min_d + np.log(np.sum(np.exp(-k * (d - min_d))))
        d_min_smooth = -log_sum_exp_neg_k_d / k
        
        if d_min_smooth < 1e-12:
            return 1e10
        
        ratio = d_max_smooth / d_min_smooth
        return ratio ** 2

    def objective_grad(pts_flat, k=50.0):
        """
        Gradient of the objective function.
        """
        pts_2d = pts_flat.reshape(N, 2)
        diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d = np.maximum(d, 1e-6)
        
        # We need grad of f = (d_max_smooth / d_min_smooth)^2
        # Let R = d_max_smooth / d_min_smooth
        # f = R^2
        # df/dpts = 2 * R * dR/dpts
        # R = A / B where A = d_max_smooth, B = d_min_smooth
        # dR/dpts = (B * dA/dpts - A * dB/dpts) / B^2
        # So df/dpts = 2 * (A/B) * (B * dA/dpts - A * dB/dpts) / B^2
        #           = 2 * A / B^2 * (B * dA/dpts - A * dB/dpts)
        
        # Compute A and B
        max_d = np.max(d)
        log_sum_exp_k_d = k * max_d + np.log(np.sum(np.exp(k * (d - max_d))))
        A = log_sum_exp_k_d / k
        
        min_d = np.min(d)
        log_sum_exp_neg_k_d = -k * min_d + np.log(np.sum(np.exp(-k * (d - min_d))))
        B = -log_sum_exp_neg_k_d / k
        
        if B < 1e-12:
            return np.zeros_like(pts_flat)
        
        # We need dA/dpts and dB/dpts.
        # A = (1/k) * log(sum(exp(k * d_i)))
        # dA/d(d_i) = (1/k) * (exp(k * d_i) * k) / sum(exp(k * d_j)) = exp(k * d_i) / sum(exp(k * d_j))
        # Let w_max_i = exp(k * (d_i - max_d)) / sum(exp(k * (d_j - max_d)))
        # Then dA/d(d_i) = w_max_i
        
        # B = -(1/k) * log(sum(exp(-k * d_i)))
        # dB/d(d_i) = -(1/k) * (exp(-k * d_i) * (-k)) / sum(exp(-k * d_j)) = exp(-k * d_i) / sum(exp(-k * d_j))
        # Let w_min_i = exp(-k * (d_i - min_d)) / sum(exp(-k * (d_j - min_d)))
        # Then dB/d(d_i) = w_min_i
        
        # Compute weights
        exp_k_d = np.exp(k * (d - max_d))
        sum_exp_k_d = np.sum(exp_k_d)
        w_max = exp_k_d / sum_exp_k_d
        
        exp_neg_k_d = np.exp(-k * (d - min_d))
        sum_exp_neg_k_d = np.sum(exp_neg_k_d)
        w_min = exp_neg_k_d / sum_exp_neg_k_d
        
        # Now we have dA/d(d_i) = w_max_i and dB/d(d_i) = w_min_i
        # We need to map this back to points.
        # d(d_i)/d(pts) is a vector.
        # For pair (i, j) with distance d_ij:
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        # d(d_ij)/d(x_j) = (x_j - x_i) / d_ij
        
        # The total gradient is:
        # grad_f = 2 * A / B^2 * (B * sum_i w_max_i * grad(d_i) - A * sum_i w_min_i * grad(d_i))
        #        = 2 * A / B^2 * (B * G_max - A * G_min)
        # where G_max = sum_i w_max_i * grad(d_i) and G_min = sum_i w_min_i * grad(d_i)
        
        # Let's compute G_max and G_min.
        # We have w_max and w_min for the upper triangle indices.
        # We need to accumulate gradients for each point.
        
        grad_G_max = np.zeros_like(pts_2d)
        grad_G_min = np.zeros_like(pts_2d)
        
        # iu[0] and iu[1] are the indices of the points in the pairs
        i_idx = iu[0]
        j_idx = iu[1]
        
        # diff for these pairs
        diff_pairs = pts_2d[i_idx] - pts_2d[j_idx]
        d_pairs = d
        
        # Unit vectors
        unit_vecs = diff_pairs / d_pairs[:, np.newaxis]
        
        # Contribution to point i: w_i * unit_vec
        # Contribution to point j: -w_i * unit_vec
        
        # Use np.add.at for accumulation
        np.add.at(grad_G_max, i_idx, w_max[:, np.newaxis] * unit_vecs)
        np.add.at(grad_G_max, j_idx, -w_max[:, np.newaxis] * unit_vecs)
        
        np.add.at(grad_G_min, i_idx, w_min[:, np.newaxis] * unit_vecs)
        np.add.at(grad_G_min, j_idx, -w_min[:, np.newaxis] * unit_vecs)
        
        # Combine
        factor = 2.0 * A / (B * B)
        grad_total = factor * (B * grad_G_max - A * grad_G_min)
        
        return grad_total.flatten()

    # Initial guess: Hexagonal lattice
    pts = np.zeros((N, 2))
    side = 4
    count = 0
    for row in range(side):
        for col in range(side):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts[count, 0] = x
            pts[count, 1] = y
            count += 1
            if count == N:
                break
        if count == N:
            break
    
    # Center and scale
    pts -= pts.mean(axis=0)
    max_d = np.max(np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2)))
    pts /= max_d
    
    best_pts = pts.copy()
    best_score = score(pts)
    
    rng = np.random.default_rng(42)
    num_restarts = 10
    
    # We will use L-BFGS-B on the smooth objective.
    # Since the problem is non-convex, we will do multiple restarts.
    # We also try different k values to control the "sharpness" of the approximation.
    k_values = [20, 50, 100]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            current_pts = best_pts.copy()
            k = 50
        else:
            # Perturb best found
            current_pts = best_pts + rng.normal(0, 0.1, (N, 2))
            current_pts -= current_pts.mean(axis=0)
            max_d = np.max(np.sqrt(np.sum((current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :])**2, axis=2)))
            current_pts /= max_d
            k = k_values[r % len(k_values)]
        
        # Run local optimization
        try:
            res = minimize(
                objective, 
                current_pts.flatten(), 
                jac=lambda x: objective_grad(x, k),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            opt_pts = res.x.reshape(N, 2)
            
            # Clean up: Center and normalize
            opt_pts -= opt_pts.mean(axis=0)
            
            sc = score(opt_pts)
            if sc < best_score:
                best_score = sc
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
