"""Packing of 50 equal circles using Differential Evolution and Subgradient Refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    if len(centres) < 2:
        min_pair = np.inf
    else:
        min_pair = pdist(centres, metric='euclidean').min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _objective(x):
    """Objective function for Differential Evolution (minimizes negative radius)."""
    centres = x.reshape(N, 2)
    return -_radius_score(centres)

def _refine(centres, iterations=100):
    """Refine the packing using subgradient ascent on the minimum distance."""
    centres = centres.copy()
    step_size = 0.01
    best_r = _radius_score(centres)
    best_centres = centres.copy()
    
    for i in range(iterations):
        # Calculate current radius
        r = _radius_score(centres)
        
        # Identify bottlenecks
        # 1. Boundary bottlenecks
        dist_bound = np.stack([
            0.5 - centres[:, 0],
            0.5 + centres[:, 0],
            0.5 - centres[:, 1],
            0.5 + centres[:, 1]
        ], axis=1)
        min_dist_bound = dist_bound.min(axis=1)
        
        # 2. Pairwise bottlenecks
        # We need the indices of the minimum pair.
        # pdist doesn't return indices directly, but we can compute the distance matrix.
        # For N=50, 50x50 matrix is small.
        dist_matrix = pdist(centres, metric='euclidean') # squared for speed? No, need sqrt for gradient direction.
        # Let's use euclidean and find min.
        # To save time, we can just use the global min from _radius_score logic if we recompute.
        # But we need the *direction* of the gradient.
        
        # Efficient way:
        # Compute all pairwise distances
        dists = pdist(centres, metric='euclidean')
        min_d = dists.min()
        
        # Find pairs with distance close to min_d (within epsilon)
        # This is expensive if done naively.
        # Alternative: Just move all points slightly away from their nearest neighbor.
        
        # Let's use a simpler force model:
        # Force on i = sum over j of (p_i - p_j) / ||p_i - p_j||^2  (Coulomb-like repulsion)
        # Plus force from walls.
        
        forces = np.zeros_like(centres)
        
        # Wall forces
        # If x < -0.5 + r, push right. If x > 0.5 - r, push left.
        # Actually, just push towards center if close to wall.
        # Wall force:
        # F_x = -sign(x) * (1/ (0.5 - |x|)^2) ?? 
        # Simpler: If dist to wall < 2*r, push away from wall.
        
        # Let's stick to the subgradient of the min function.
        # The gradient of min(d1, d2, ...) is the gradient of the minimum term.
        # If multiple terms are minimum, it's the sum of their gradients.
        
        # 1. Boundary terms
        # For each point i, check which wall is closest.
        # d_left = 0.5 + x_i
        # d_right = 0.5 - x_i
        # d_bottom = 0.5 + y_i
        # d_top = 0.5 - y_i
        
        d_left = 0.5 + centres[:, 0]
        d_right = 0.5 - centres[:, 0]
        d_bottom = 0.5 + centres[:, 1]
        d_top = 0.5 - centres[:, 1]
        
        # Find min wall dist for each point
        wall_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
        min_wall_idx = wall_dists.argmin(axis=1)
        min_wall_dist = wall_dists.min(axis=1)
        
        # Add wall forces
        # If min_wall_dist is part of the global minimum, we should move away.
        # But we don't know which ones are global minimum without comparing to pair min.
        
        # Global r is min( min_pair/2, min_wall )
        # Let's just calculate the gradient contribution for all "tight" constraints.
        
        # This is getting complex to implement correctly and fast in a few lines.
        # A simpler local search: Random perturbations with momentum (like the original SA but with momentum and adaptive step).
        
        # Let's revert to a simpler local polish: 
        # Try small random moves, accept if better. (This is just SA again).
        
        # To be distinct, let's use **Nelder-Mead** on a subset? No.
        
        # Let's just return the DE result. DE with `polish=True` is risky due to timeout.
        # I will use DE with `polish=False` and a simple **Gradient Step** based on the nearest neighbor repulsion.
        
        # Force from neighbors:
        # For each i, find nearest neighbor j. Move i away from j.
        # This is O(N^2).
        
        # Compute distance matrix
        # Using scipy.spatial.distance_matrix or just broadcasting
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Nearest neighbor index
        nn_idx = np.argmin(dists, axis=1)
        nn_dist = dists[np.arange(N), nn_idx]
        
        # Direction
        # dir_i = (p_i - p_nn) / dist
        nn_points = centres[nn_idx]
        dirs = centres - nn_points
        dirs = dirs / (nn_dist[:, np.newaxis] + 1e-9)
        
        # Wall direction
        # Push away from nearest wall
        wall_dirs = np.zeros_like(centres)
        for i in range(N):
            # Check 4 walls
            # Left: 0.5 + x
            # Right: 0.5 - x
            # Bottom: 0.5 + y
            # Top: 0.5 - y
            # Find min
            dl = 0.5 + centres[i, 0]
            dr = 0.5 - centres[i, 0]
            db = 0.5 + centres[i, 1]
            dt = 0.5 - centres[i, 1]
            min_w = min(dl, dr, db, dt)
            
            if min_w == dl: wall_dirs[i] = [1, 0] # Push right
            elif min_w == dr: wall_dirs[i] = [-1, 0] # Push left
            elif min_w == db: wall_dirs[i] = [0, 1] # Push up
            else: wall_dirs[i] = [0, -1] # Push down
            
        # Combine forces
        # We want to increase the minimum distance.
        # Move in direction of nearest neighbor repulsion AND wall repulsion.
        # Scale by step size.
        
        new_centres = centres + step_size * (dirs + wall_dirs)
        
        # Clamp to bounds
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_r = _radius_score(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            best_centres = new_centres
            centres = new_centres
            # Increase step size slightly if improving? Or keep constant.
        else:
            # Reduce step size
            step_size *= 0.9
            # Revert
            # centres remains unchanged
            
    return best_centres

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Run Differential Evolution
    # maxiter=50, popsize=15 to stay within time limits
    # seed=42 for reproducibility in this context, but random is fine too.
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False
    )
    
    best_x = result.x
    best_centres = best_x.reshape(N, 2)
    
    # Refine with subgradient/force-based local search
    best_centres = _refine(best_centres, iterations=100)
    
    return best_centres
# EVOLVE-BLOCK-END
