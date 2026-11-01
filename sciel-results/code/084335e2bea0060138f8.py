"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """Construct centres of N equal circles inside [-1/2,1/2]^2 to maximize radius."""
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Initial grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    def compute_radius(c):
        """Compute the valid radius for a given configuration."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        dist_to_bound = np.min(0.5 - np.abs(c), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        
        r = min(min_pair_dist / 2.0, min_bound_dist)
        return r if r > 0 else 1e-15
    
    def score(c):
        """Negative radius for minimization."""
        r = compute_radius(c)
        return -r
    
    # Multi-stage optimization with annealing
    best_centers = centers.copy()
    best_score = score(best_centers)
    
    # Stage 1: Moderate temperature
    temps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for temp in temps:
        if time.time() > deadline - 30:
            break
            
        # Add small perturbation
        perturbed = best_centers + np.random.randn(*best_centers.shape) * temp
        # Keep within bounds
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Use Nelder-Mead for local optimization
        try:
            result = minimize(
                score,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
        
        # Additional polish with even tighter tolerances
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                score,
                best_centers,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                score,
                best_centers,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    final_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return final_centers
# EVOLVE-BLOCK-END
