import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave margin for safety
    
    best_c = None
    best_r = 0.0
    
    # Helper to calculate the actual radius (min distance)
    def calc_radius(c):
        c = c.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        idx = np.where(np.triu(np.ones(dists.shape, dtype=bool), k=1))
        min_pair = np.min(dists[idx]) / 2.0
        min_wall = 0.5 - np.max(np.abs(c), axis=1)
        min_wall_val = np.min(min_wall)
        return min(min_pair, min_wall_val)

    # Soft min objective to minimize
    def objective(c_flat, lam):
        c = c_flat.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        m = np.min(all_dists)
        if m < 0:
            return 1000.0 - m
        
        vals = np.exp(-lam * (all_dists - m))
        soft_min_val = m - (1.0/lam) * np.log(np.sum(vals))
        return -soft_min_val

    # Potential field objective: sum of inverse-power repulsions
    def potential_objective(c_flat, p_power, wall_coeff):
        c = c_flat.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Pairwise repulsion: sum 1/d^p for i<j
        # Use upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[iu]
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        pair_potential = np.sum(1.0 / np.power(pair_dists_sq + eps, p_power / 2.0))
        
        # Wall repulsion: for each sphere, distance to nearest wall
        # d_wall_i = 0.5 - max(|c_ik|) for k in {0,1,2}
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        wall_potential = wall_coeff * np.sum(1.0 / np.power(np.maximum(wall_dists, eps), p_power))
        
        return pair_potential + wall_potential

    # Gradient of potential field
    def potential_gradient(c_flat, p_power, wall_coeff):
        c = c_flat.reshape(N, 3)
        grad = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Pairwise gradient: d/dx_i (1/|x_i - x_j|^p) = -p * (x_i - x_j) / |x_i - x_j|^(p+2)
        iu = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[iu]
        eps = 1e-8
        
        # For each pair (i,j) with i<j:
        # grad_i += -p * (c_i - c_j) / (d_ij^2 + eps)^(p/2 + 1) * d_ij^p... 
        # Actually: d/dx (1/|x|^p) = -p * x / |x|^(p+2)
        # So for pair (i,j): contribution to grad[i] is -p * (c[i]-c[j]) / (|c[i]-c[j]|^2)^(p/2+1) * |c[i]-c[j]|^p... 
        # Let me be more careful:
        # f = 1/|x|^p where x = c_i - c_j
        # df/dx_i = -p * x_i / |x|^(p+2)
        # In terms of components: grad[i] += -p * (c[i] - c[j]) / (|c[i]-c[j]|^2)^(p/2+1)
        
        # Vectorized: for each pair (i,j), compute factor = -p / (d_ij^2 + eps)^(p/2 + 1)
        # Then grad[i] += factor * (c[i] - c[j])
        #      grad[j] -= factor * (c[i] - c[j])  (by Newton's third law)
        
        factors = -p_power / np.power(pair_dists_sq + eps, p_power/2.0 + 1.0)
        
        # Apply to gradient
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            f = factors[idx]
            diff_ij = c[i] - c[j]
            grad[i] += f * diff_ij
            grad[j] -= f * diff_ij
        
        # Wall gradient: d/dx_i (1/(0.5 - |x_ik|)^p) for the k with max |x_ik|
        # For each sphere i, find which coordinate k has max |c_ik|
        # Then d/dc_ik [1/(0.5 - |c_ik|)^p] = p * sign(c_ik) / (0.5 - |c_ik|)^(p+1)
        abs_c = np.abs(c)
        max_abs_idx = np.argmax(abs_c, axis=1)  # shape (N,)
        max_abs_vals = np.max(abs_c, axis=1)     # shape (N,)
        wall_dists = 0.5 - max_abs_vals
        wall_dists_clipped = np.maximum(wall_dists, eps)
        
        # Gradient contribution from walls
        wall_grad_factors = wall_coeff * p_power / np.power(wall_dists_clipped, p_power + 1.0)
        
        # Apply to correct coordinate
        for i in range(N):
            k = max_abs_idx[i]
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0  # arbitrary if exactly 0
            grad[i, k] += wall_grad_factors[i] * sign
        
        return grad.flatten()

    # Gradient descent with adaptive step size for potential field
    def potential_field_optimize(c0, p_power, wall_coeff, max_iters=500, base_lr=0.01):
        c = c0.copy()
        lr = base_lr
        prev_pot = potential_objective(c.flatten(), p_power, wall_coeff)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            grad = potential_gradient(c.flatten(), p_power, wall_coeff)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                break
            
            # Adaptive step: reduce if potential increases
            c_new = c - lr * grad.reshape(N, 3)
            # Clip to stay within bounds
            c_new = np.clip(c_new, -0.49, 0.49)
            
            new_pot = potential_objective(c_new.flatten(), p_power, wall_coeff)
            
            if new_pot < prev_pot:
                c = c_new
                prev_pot = new_pot
                # Slightly increase learning rate if improving
                lr = min(lr * 1.05, 0.1)
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        return c

    # Polish using L-BFGS-B with soft-min
    def polish(c_init, max_time_remaining=30.0):
        bounds = [(-0.5, 0.5)] * (N * 3)
        current_c = c_init.flatten()
        
        # Try increasing lambda for sharper min
        lams = [50.0, 200.0, 500.0]
        
        for lam in lams:
            remaining = time_limit - (time.time() - start_time)
            if remaining < 5:
                break
            
            try:
                res = minimize(
                    objective, 
                    current_c, 
                    args=(lam,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_c = res.x
            except Exception:
                pass
        
        return current_c.reshape(N, 3)

    n_restarts = 15
    
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 20:  # Reserve time for polishing
            break
            
        # Random initialization
        c0 = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Phase 1: Potential field optimization
        p_powers = [4.0, 6.0]
        wall_coeffs = [2.0, 5.0]
        
        best_phase1_c = c0
        best_phase1_r = calc_radius(c0)
        
        for pp in p_powers:
            for wc in wall_coeffs:
                if time.time() - start_time > time_limit - 15:
                    break
                
                c_opt = potential_field_optimize(c0, pp, wc, max_iters=300, base_lr=0.02)
                r_opt = calc_radius(c_opt)
                
                if r_opt > best_phase1_r:
                    best_phase1_r = r_opt
                    best_phase1_c = c_opt
        
        # Phase 2: Polish with soft-min L-BFGS
        if time.time() - start_time < time_limit - 10:
            c_polished = polish(best_phase1_c)
            r_polished = calc_radius(c_polished)
            
            if r_polished > best_phase1_r:
                best_phase1_c = c_polished
                best_phase1_r = r_polished
        
        if best_phase1_r > best_r:
            best_r = best_phase1_r
            best_c = best_phase1_c

    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_c
