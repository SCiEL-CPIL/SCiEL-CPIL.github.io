import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _soft_min_objective(c, beta):
    """
    Compute the negative soft-min of distances.
    c: shape (N, 3)
    beta: temperature parameter
    Returns: objective value to minimize.
    """
    # Reshape c to (N, 1, 3) for broadcasting
    c = c.reshape(N, 1, 3)
    
    # Pairwise distances
    # diff shape (N, N, 3)
    diff = c - c.T
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal to avoid zero distance (self-interaction)
    mask = ~np.eye(N, dtype=bool)
    dists_sq = np.where(mask, dists_sq, np.inf)
    
    # Wall distances
    # c shape (N, 3)
    wall_dists = 0.5 - np.abs(c.reshape(N, 3))
    # wall_dists shape (N, 3)
    
    # We want min of (dists/2) and wall_dists
    # Let's collect all distances into a single array for log-sum-exp
    # To save memory, we can compute log-sum-exp in parts or just flatten
    
    # Flatten pair distances
    pair_dists = np.sqrt(dists_sq[mask]) / 2.0
    
    # Flatten wall distances
    wall_dists_flat = wall_dists.flatten()
    
    # Combine
    # To avoid overflow in exp, we use the log-sum-exp trick:
    # log(sum(exp(-beta * x))) = -beta * min(x) + log(sum(exp(-beta * (x - min(x)))))
    # We want to minimize: (1/beta) * log(sum(exp(-beta * x)))
    # This is approximately min(x) for large beta.
    
    # Let's just compute the values.
    # If beta is large, exp(-beta * x) underflows for large x, which is good.
    # But if beta is small, it's fine.
    
    # To be safe and efficient:
    x_pair = -beta * pair_dists
    x_wall = -beta * wall_dists_flat
    
    # Combine into one array?
    # N=100, pairs ~ 5000, walls 300. Total ~ 5300 values.
    # This is small enough to combine.
    x_all = np.concatenate([x_pair, x_wall])
    
    # Log-sum-exp stability
    x_max = np.max(x_all)
    if x_max == -np.inf:
        return 0.0
    s = np.sum(np.exp(x_all - x_max))
    log_sum_exp = x_max + np.log(s)
    
    # Objective is - (1/beta) * log_sum_exp
    # Wait, soft_min = -1/beta * log(sum(exp(-beta x)))
    # We want to MAXIMIZE soft_min.
    # So we MINIMIZE -soft_min = 1/beta * log(sum(exp(-beta x)))
    obj = log_sum_exp / beta
    return obj

def _compute_radius(c):
    """Compute the actual radius for a configuration."""
    c = np.clip(c, -0.5, 0.5)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq)) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialization: 5x5x4 grid
    # x, y: 5 points, z: 4 points
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    
    c = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    # c shape (100, 3)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Optimization loop with increasing beta
    # Start with small beta, increase gradually
    betas = [10, 20, 50, 100, 200, 500]
    
    current_c = c.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Define objective function for this beta
        def objective(c_flat):
            c_reshaped = c_flat.reshape(N, 3)
            return _soft_min_objective(c_reshaped, beta)
        
        # Use L-BFGS-B
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective, 
                current_c.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 5000}
            )
            
            if res.success or res.fun < objective(current_c.flatten()):
                new_c = res.x.reshape(N, 3)
                r = _compute_radius(new_c)
                if r > best_r:
                    best_r = r
                    best_c = new_c.copy()
                current_c = new_c
        except Exception:
            pass
            
    # Final polish with a very high beta or Nelder-Mead on the best result?
    # Let's try one more step with high beta if time permits
    if time.time() < deadline - 5:
        beta_final = 1000
        def objective_final(c_flat):
            c_reshaped = c_flat.reshape(N, 3)
            return _soft_min_objective(c_reshaped, beta_final)
            
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                objective_final, 
                best_c.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 2000}
            )
            new_c = res.x.reshape(N, 3)
            r = _compute_radius(new_c)
            if r > best_r:
                best_r = r
                best_c = new_c.copy()
        except Exception:
            pass
            
    return best_c

if __name__ == "__main__":
    c = construct()
    print("Radius:", _compute_radius(c))
