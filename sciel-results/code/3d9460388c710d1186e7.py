"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum radius.
    
    Strategy:
    1. Start with a good initial configuration (FCC-inspired or grid-based).
    2. Use simulated annealing with local optimization to improve the packing.
    3. The objective is to maximize the minimum distance between centers and to walls.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s safety margin
    
    # Initialize with a grid-based configuration, then perturb
    m = 5  # 5x5x5 = 125 positions, take first 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    def compute_radius(c):
        """Compute the largest common radius for given centers."""
        # Minimum distance between any two centers
        from scipy.spatial.distance import pdist
        d = pdist(c, metric='euclidean')
        min_dist = d.min() if len(d) > 0 else 1.0
        
        # Minimum distance to walls
        wall_dists = np.abs(c).min(axis=1)
        min_wall = (0.5 - np.abs(c).max(axis=1)).min()
        
        return min(min_dist / 2.0, min_wall)
    
    def objective(c):
        """Negative of the radius (we minimize negative radius)."""
        return -compute_radius(c)
    
    current_radius = compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Simulated annealing
    T = 0.05  # Initial temperature
    cooling = 0.9995
    n_iterations = 50000
    
    for i in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random center to move
        idx = rng.randint(N)
        old_pos = centers[idx].copy()
        
        # Random perturbation
        delta = rng.normal(0, T * 0.5, size=3)
        new_pos = old_pos + delta
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        # Compute new radius
        centers[idx] = new_pos
        new_radius = compute_radius(centers)
        
        # Accept if better, or with probability based on temperature
        if new_radius >= current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Boltzmann acceptance
            delta_e = current_radius - new_radius
            if rng.random() < np.exp(-delta_e / T):
                current_radius = new_radius
        
        # Cool down
        T *= cooling
        
        # If temperature is too low, restart with higher temperature
        if T < 0.001:
            T = 0.02
            # Add small perturbation to escape local minima
            centers += rng.uniform(-0.005, 0.005, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            current_radius = compute_radius(centers)
    
    # Final local optimization using Nelder-Mead
    if time.time() < deadline - 10:
        # Optimize subsets of centers locally
        for start in range(0, N, 10):
            if time.time() > deadline - 5:
                break
            end = min(start + 10, N)
            
            def sub_objective(sub_c):
                full_c = best_centers.copy()
                full_c[start:end] = sub_c
                return -compute_radius(full_c)
            
            x0 = best_centers[start:end].copy()
            result = minimize(sub_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < -best_radius:
                best_centers[start:end] = result.x
                best_radius = -result.fun
    
    return best_centers
# EVOLVE-BLOCK-END
