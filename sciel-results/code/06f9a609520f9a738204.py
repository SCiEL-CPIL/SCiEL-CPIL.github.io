"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
import math

def _initial_guess():
    """Create a good initial configuration using a grid with some perturbation."""
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbations to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    return centers

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(min_boundary)
    
    if n < 2:
        return min_boundary
    
    # Minimum pairwise distance
    min_pairwise = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pairwise:
                min_pairwise = d
    
    return min(min_boundary, min_pairwise / 2.0)

def _objective(centers):
    """Objective function to minimize (negative of radius)."""
    r = _compute_radius(centers)
    return -r

def construct():
    """Construct the best packing found within time budget."""
    start_time = time.time()
    deadline = 145.0  # 145 seconds to leave safety margin
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    best_centers = _initial_guess()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple initial configurations and local optimizations
    for attempt in range(20):
        if time.time() - start_time > deadline - 30:
            break
        
        # Generate initial guess
        if attempt == 0:
            centers = _initial_guess()
        else:
            # Use perturbed version of best found so far
            centers = best_centers + np.random.uniform(-0.05, 0.05, size=(N, 2))
            # Clip to valid region
            centers = np.clip(centers, -0.49, 0.49)
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(
                _objective,
                centers,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < -best_radius:
                new_radius = -result.fun
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = result.x.copy()
        except Exception:
            pass
        
        # If we have time, try a more precise polish
        if time.time() - start_time > deadline - 10:
            break
        
        # Try BFGS as a second pass
        try:
            result2 = minimize(
                _objective,
                best_centers,
                method='BFGS',
                options={
                    'maxiter': 5000,
                    'gtol': 1e-10
                }
            )
            
            if result2.fun < -best_radius:
                new_radius = -result2.fun
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = result2.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerances if time permits
    if time.time() - start_time < deadline - 5:
        try:
            result_final = minimize(
                _objective,
                best_centers,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result_final.fun < -best_radius:
                best_radius = -result_final.fun
                best_centers = result_final.x.copy()
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
