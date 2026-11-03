"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    def min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_centers = centers.copy()
    best_r = min_radius(centers)
    
    # Soft-min objective using log-sum-exp (vectorized)
    def soft_min_radius(x, beta):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        pair_half = d / 2.0
        all_vals = np.concatenate([pair_half[np.triu_indices(N, k=1)], dist_to_walls])
        
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        log_sum = np.log(np.sum(exp_vals))
        soft_min = max_val - log_sum / beta
        
        return -soft_min
    
    # Vectorized gradient computation
    def soft_min_radius_grad(x, beta):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        pair_half = d / 2.0
        all_vals = np.concatenate([pair_half[np.triu_indices(N, k=1)], dist_to_walls])
        
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        weights = exp_vals / np.sum(exp_vals)
        
        n_pairs = N * (N - 1) // 2
        pair_weights = weights[:n_pairs]
        wall_weights = weights[n_pairs:]
        
        grad_x = np.zeros_like(x)
        
        # Vectorized pair contributions
        # For each pair (i,j), contribution to grad is w_ij * (c_i - c_j) / |c_i - c_j|
        # We need to accumulate: grad[i] += sum_j w_ij * (c_i - c_j) / d_ij
        #                     grad[j] -= sum_i w_ij * (c_i - c_j) / d_ij
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        # Unit vectors for each pair
        vecs = c[i_idx] - c[j_idx]  # shape (n_pairs, 3)
        norms = d[i_idx, j_idx]  # shape (n_pairs,)
        
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-9, norms, 1.0)
        unit_vecs = vecs / safe_norms[:, np.newaxis]
        
        # Weighted unit vectors
        weighted_unit_vecs = pair_weights[:, np.newaxis] * unit_vecs
        
        # Accumulate gradients using np.add.at
        for k in range(3):
            np.add.at(grad_x[:, k], i_idx, weighted_unit_vecs[:, k])
            np.add.at(grad_x[:, k], j_idx, -weighted_unit_vecs[:, k])
        
        # Wall contributions (vectorized)
        # For each sphere i, wall distance = min_k (0.5 - |c[i,k]|)
        # Gradient: if c[i,k] > 0, grad wrt c[i,k] = -1; else +1
        # The active wall is the one with minimum distance
        
        # Determine which wall is active for each sphere
        # dist_to_walls[i] = min_k (0.5 - |c[i,k]|)
        # We need to know which k gives the minimum
        
        abs_c = np.abs(c)
        wall_dists = 0.5 - abs_c  # shape (N, 3)
        active_wall = np.argmin(wall_dists, axis=1)  # shape (N,)
        
        # Sign: if c[i, active_wall[i]] > 0, sign = -1; else +1
        active_vals = c[np.arange(N), active_wall]
        signs = np.where(active_vals > 0, -1.0, 1.0)
        
        # Add wall gradient contributions
        for i in range(N):
            if wall_weights[i] > 1e-15:
                grad_x[i * 3 + active_wall[i]] += wall_weights[i] * signs[i]
        
        return -grad_x
    
    # Continuation method: ramp up beta
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    x = centers.flatten()
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
        
        # L-BFGS-B optimization
        try:
            remaining = time_limit - 10 - elapsed
            maxiter = min(300, max(50, int(remaining * 10)))
            res = minimize(
                fun=lambda x: soft_min_radius(x, beta),
                x0=x,
                jac=lambda x: soft_min_radius_grad(x, beta),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
        
        # Update best
        c = x.reshape(N, 3)
        r = min_radius(c)
        if r > best_r:
            best_r = r
            best_centers = c.copy()
    
    # Final polish with higher beta
    elapsed = time.time() - start_time
    if elapsed < time_limit - 8:
        beta_final = 5000
        try:
            remaining = time_limit - 5 - elapsed
            maxiter = min(800, max(100, int(remaining * 15)))
            res = minimize(
                fun=lambda x: soft_min_radius(x, beta_final),
                x0=x,
                jac=lambda x: soft_min_radius_grad(x, beta_final),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            x = res.x
            c = x.reshape(N, 3)
            r = min_radius(c)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        except Exception:
            pass
    
    # Second polish if time remains
    elapsed = time.time() - start_time
    if elapsed < time_limit - 3:
        beta_final2 = 10000
        try:
            remaining = time_limit - 2 - elapsed
            maxiter = min(500, max(50, int(remaining * 10)))
            res = minimize(
                fun=lambda x: soft_min_radius(x, beta_final2),
                x0=x,
                jac=lambda x: soft_min_radius_grad(x, beta_final2),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 100}
            )
            x = res.x
            c = x.reshape(N, 3)
            r = min_radius(c)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
