"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5


def _compute_score(centers):
    """Compute the packing radius for a set of centers."""
    # Clip to valid range
    centers = np.clip(centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    # Wall distance
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for N=50 (50x50 is fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _soft_min_obj(params, beta, centers_shape=(N, 3)):
    """
    Objective: minimize -soft_min(distances)
    where distances include both pairwise distances/2 and wall distances.
    We want to maximize the minimum distance, so we minimize the negative.
    """
    centers = params.reshape(centers_shape)
    # Clip to stay inside
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centers)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Collect all distances
    all_dists = np.concatenate([wall_dists.flatten(), dists[np.triu_indices(N, k=1)]])
    
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * d)))
    # For numerical stability
    min_d = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - np.log(np.sum(exp_vals)) / beta
    
    # We want to maximize soft_min, so minimize negative
    return -soft_min_val


def _build_initial_grid():
    """
    Build a good initial configuration.
    Try a body-centered cubic-like arrangement or a perturbed grid.
    """
    # For 50 spheres, a 4x4x4 grid gives 64 spots, we need 50.
    # Let's use a 4x4x4 grid but remove corners or use a different layout.
    # Actually, let's try a 3x4x5 = 60 grid, remove 10.
    # Or better: use a FCC-like structure.
    
    # Simple approach: 4x4x4 grid, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    perturbation = 0.01 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 1e-3, CUBE_HALF - 1e-3)
    
    return grid[:N]


def construct():
    """
    Multi-phase optimization:
    1. Start from a perturbed grid
    2. Use continuation method: start with low beta (smooth), increase beta
    3. Use L-BFGS-B for optimization
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    centers = _build_initial_grid()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Continuation schedule for beta
    # Start with low beta (smooth approximation) and increase
    beta_schedule = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        # Optimize with current beta
        try:
            result = minimize(
                _soft_min_obj,
                centers.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers
        except Exception:
            pass
    
    # Final local refinement with high beta
    if time.time() < deadline - 5.0:
        try:
            for beta in [5000, 10000]:
                if time.time() > deadline - 5.0:
                    break
                result = minimize(
                    _soft_min_obj,
                    centers.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    centers = new_centers
        except Exception:
            pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers


# EVOLVE-BLOCK-START
# The main construct() function is defined above
# EVOLVE-BLOCK-END
