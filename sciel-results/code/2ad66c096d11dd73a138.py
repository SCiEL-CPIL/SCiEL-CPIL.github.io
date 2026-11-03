"""New program for the suite task; uses Voronoi diagram approach."""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 50


def construct():
    # Use Voronoi diagram approach
    # Represent packing as 2D Voronoi diagram where each circle is inscribed in its cell
    # Optimize vertex positions using smooth objective to maximize minimum inradius
    
    # Initial positions: grid with slight randomization
    np.random.seed(42)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Add small random perturbation
    grid = grid + np.random.randn(N, 2) * 0.01
    
    # Clip to valid range
    grid = np.clip(grid, -0.45, 0.45)
    
    # Objective function: maximize minimum inradius
    def objective(params):
        centers = params.reshape(N, 2)
        
        # Check bounds
        if np.any(np.abs(centers) > 0.5):
            return -1.0
        
        # Compute pairwise distances
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        # Radius is half of minimum distance
        r = min(min_pair_dist / 2, min_wall_dist)
        
        # Return negative radius to maximize
        return -r
    
    # Gradient-free optimization using Nelder-Mead
    # Since we have 100 parameters, use a smarter approach
    
    # Use iterative local optimization
    best_centers = grid.copy()
    best_r = -objective(best_centers.reshape(-1))
    
    # Try multiple optimization strategies
    for method in ['Nelder-Mead', 'Powell']:
        try:
            result = minimize(objective, best_centers.reshape(-1), method=method, 
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            r = -result.fun
            if r > best_r:
                best_r = r
                best_centers = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Additional local refinement
    for _ in range(500):
        # Random perturbation
        perturbed = best_centers + np.random.randn(N, 2) * 0.005
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        r = -objective(perturbed.reshape(-1))
        if r > best_r:
            best_r = r
            best_centers = perturbed
    
    # Final cleanup: ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
