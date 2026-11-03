"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit
    
    # 1. Initialization: Jittered Grid
    # A simple grid is often a local minimum. Jittering helps escape.
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # m will be 4 for N=50 (4^3=64 >= 50)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_r = 0.0
    
    # Helper to compute exact min radius
    def min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    # Soft-min function for smooth optimization
    # softmin(z) = -1/beta * ln(sum(exp(-beta * z)))
    # We want to maximize the min distance, so we minimize -softmin(distances)
    def objective(x, beta):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        pair_dists = d / 2.0
        
        # Wall distances
        dist_to_walls = np.minimum(c + 0.5, 0.5 - c)
        wall_dists = np.min(dist_to_walls, axis=1)
        
        # Combine all distances
        # We need to be careful with exp overflow/underflow.
        # z = distances. We want -1/beta * ln(sum(exp(-beta * z)))
        # Let M = min(z). Then sum(exp(-beta*z)) = exp(-beta*M) * sum(exp(-beta*(z-M)))
        # softmin = M - 1/beta * ln(sum(exp(-beta*(z-M))))
        
        all_dists = np.concatenate([pair_dists.ravel(), wall_dists.ravel()])
        
        # Filter out inf/nans if any (shouldn't happen if centers are valid)
        all_dists = all_dists[~np.isnan(all_dists)]
        
        if len(all_dists) == 0:
            return 0.0
            
        M = np.min(all_dists)
        # Avoid overflow in exp
        exp_terms = np.exp(-beta * (all_dists - M))
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            soft_min_val = M
        else:
            soft_min_val = M - (1.0 / beta) * np.log(sum_exp)
            
        return -soft_min_val

    # Phase 1: Smooth Optimization with Annealing
    # Start with low beta (smooth), increase to high beta (sharp min)
    betas = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    current_x = centers.flatten()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 20:
            break
            
        def obj_func(x):
            return objective(x, beta)
            
        try:
            result = minimize(
                obj_func,
                current_x,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.x is not None:
                new_centers = result.x.reshape(N, 3)
                # Clip to ensure valid
                new_centers = np.clip(new_centers, -0.499, 0.499)
                
                # Check if this is better than current best using exact metric
                current_r = min_radius(new_centers)
                if current_r > best_r:
                    best_r = current_r
                    best_centers = new_centers.copy()
                
                # Update current_x for next iteration
                current_x = new_centers.flatten()
                
        except Exception:
            pass

    # Phase 2: Final Polish with Non-Smooth Subgradient/Local Search
    # Use the best smooth result as a starting point for a tighter local search
    centers = best_centers.copy()
    current_r = best_r
    
    # A few iterations of subgradient ascent to fix any "soft" errors
    T = 0.001
    cooling_rate = 0.95
    min_T = 1e-5
    max_iter = 1000
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Calculate subgradient
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        G = np.zeros_like(centers)
        eps = 1e-5
        
        # Only consider pairs/walls close to the minimum to save time and focus
        # Threshold: 2*r + eps
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                if d[i_idx, j_idx] < 2 * r + eps:
                    vec = centers[i_idx] - centers[j_idx]
                    norm = d[i_idx, j_idx]
                    if norm > 1e-9:
                        G[i_idx] += vec / norm
                        G[j_idx] -= vec / norm
            
            if dist_to_walls[i_idx] < r + eps:
                for k in range(3):
                    if centers[i_idx, k] > 0:
                        G[i_idx, k] -= 1.0
                    else:
                        G[i_idx, k] += 1.0
        
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-9:
            G /= norm_G
            
        # Step size decays
        step_size = 0.001 * (T / 0.001) + 0.0001
        noise = np.random.normal(0, 0.001, centers.shape)
        new_centers = centers + step_size * G + noise
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = min_radius(new_centers)
        
        if new_r > r or np.random.rand() < np.exp(-(r - new_r) / T):
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
                
        T *= cooling_rate
        if T < min_T:
            break

    return best_centers
# EVOLVE-BLOCK-END
