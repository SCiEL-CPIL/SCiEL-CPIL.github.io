"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    deadline = time.time() + 130  # 20s safety margin before 150s kill
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    def exact_radius(c):
        """Compute the exact minimum radius for configuration c."""
        c = c.reshape(N, 3)
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        # Pairwise distances using cdist for efficiency
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(np.min(wall_dists), min_pair_dist / 2.0)
    
    def soft_min_radius(c, beta=20.0):
        """
        Smooth approximation of the radius using soft-min.
        Uses log-sum-exp for stable computation.
        """
        c = c.reshape(N, 3)
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists, axis=1)  # shape (N,)
        
        # Pairwise distances - only compute upper triangle for efficiency
        # Use cdist but take only min off-diagonal
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        # Soft-min of pairwise distances / 2
        # Use log-sum-exp trick: soft_min(x) = -log(sum(exp(-x/beta))) * beta
        # But we want the minimum, so: soft_min = -logsumexp(-x/beta) * beta
        
        # For stability, subtract max
        pair_vals = dists / 2.0
        # Only consider off-diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_vals_flat = pair_vals[mask]
        
        if len(pair_vals_flat) == 0:
            soft_pair = np.inf
        else:
            # Soft-min via log-sum-exp
            neg_vals = -pair_vals_flat / beta
            max_neg = np.max(neg_vals)
            soft_min_pair = -beta * (max_neg + np.log(np.sum(np.exp(neg_vals - max_neg))))
        
        # Soft-min of wall distances
        neg_wall = -min_wall / beta
        max_neg_wall = np.max(neg_wall)
        soft_min_wall = -beta * (max_neg_wall + np.log(np.sum(np.exp(neg_wall - max_neg_wall))))
        
        # Combine: soft-min of (soft_min_wall, soft_min_pair)
        # Actually, the radius is min(min_wall, min_pair_dist/2)
        # So we need soft-min of the two aggregate values
        vals = np.array([soft_min_wall, soft_min_pair])
        neg_comb = -vals / beta
        max_neg_comb = np.max(neg_comb)
        result = -beta * (max_neg_comb + np.log(np.sum(np.exp(neg_comb - max_neg_comb))))
        
        return result
    
    def neg_soft_radius(c, beta=20.0):
        return -soft_min_radius(c, beta)
    
    def neg_soft_radius_grad(c, beta=20.0):
        """
        Gradient of -soft_min_radius.
        This is complex, so we'll use numerical gradient or just let L-BFGS-B estimate it.
        For now, return None to let scipy estimate.
        """
        return None
    
    best_x = x0.copy()
    best_r = exact_radius(x0)
    
    # Multiple restarts with smooth optimization, then polish
    max_restarts = 8
    restart_count = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution
            noise_scale = 0.03 * (1.0 + 0.5 * restart)
            x_start = best_x + np.random.randn(N, 3) * noise_scale
            x_start = np.clip(x_start, -0.49, 0.49)
        
        # Use smooth objective for optimization
        try:
            # Start with higher beta (smoother), then decrease
            beta = max(5.0, 20.0 - 2.0 * restart)
            
            result = minimize(
                neg_soft_radius,
                x_start,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 150, 'maxls': 40, 'ftol': 1e-8}
            )
            
            x_opt = result.x
            r_opt = exact_radius(x_opt)
            
            if r_opt > best_r + 1e-12:
                best_r = r_opt
                best_x = x_opt.copy()
                restart_count = 0
            else:
                restart_count += 1
                
        except Exception:
            restart_count += 1
        
        if time.time() > deadline:
            break
    
    # Final polish: use exact objective with tighter tolerance
    if time.time() < deadline - 5:
        try:
            def neg_exact_radius(c):
                return -exact_radius(c)
            
            result = minimize(
                neg_exact_radius,
                best_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'maxls': 30, 'ftol': 1e-10}
            )
            
            x_polished = result.x
            r_polished = exact_radius(x_polished)
            
            if r_polished > best_r + 1e-12:
                best_r = r_polished
                best_x = x_polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
