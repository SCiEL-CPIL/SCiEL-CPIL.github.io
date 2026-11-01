"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Best known radius from Specht: ~0.1113823475125
    # We aim to beat the simple cubic grid (which gives ~0.111111...)
    # by using local optimization (L-BFGS-B) with a smooth objective.

    def compute_radius(c):
        """Compute the minimum radius for a given set of centers."""
        # Distance between all pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(c):
        """Negative radius to maximize."""
        return -compute_radius(c)

    def objective_smooth(c, mu=0.05):
        """Smooth approximation of the negative radius using log-sum-exp."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Smooth min of squared distances
        min_d_sq = -mu * np.log(np.sum(np.exp(-dists_sq / mu)))
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = -mu * np.log(np.sum(np.exp(-wall_dists / mu)))
        
        # Combine: we want min(min_d/2, min_wall)
        # Approximate: smooth min of (min_d_sq/4) and (min_wall^2)
        val1 = min_d_sq / 4.0
        val2 = min_wall ** 2
        combined = -mu * np.log(np.exp(-val1 / mu) + np.exp(-val2 / mu))
        
        return -np.sqrt(combined)

    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    c0 = c0 + rng.uniform(-0.01, 0.01, size=c0.shape)
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_c = c0.copy()
    best_r = compute_radius(best_c)
    
    # Multi-start optimization
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 20.0:
            break
            
        if i == 0:
            c_init = c0.copy()
        else:
            # Random restarts with larger perturbations
            c_init = best_c + rng.uniform(-0.05, 0.05, size=(N, 3))
            # Clip to bounds
            c_init = np.clip(c_init, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Use smooth objective for faster convergence
        try:
            res = minimize(
                objective_smooth,
                c_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            c_opt = res.x.reshape(N, 3)
            r_opt = compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Final refinement with exact objective on best solution
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000}
            )
            c_opt = res.x.reshape(N, 3)
            r_opt = compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_c
# EVOLVE-BLOCK-END
