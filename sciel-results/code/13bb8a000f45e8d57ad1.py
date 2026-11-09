"""Improved packing for 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Use a body-centered cubic (BCC) lattice arrangement which packs better than simple cubic.
    # BCC has a packing density of ~68% vs ~52% for simple cubic.
    # We'll generate points from a BCC lattice and select the best N=100 subset,
    # then optimize with a repulsion-based energy minimization.
    
    # First, generate candidate positions from a BCC-like structure
    # In a BCC lattice, atoms are at corners and center of each cube cell.
    # For a unit cube [-0.5, 0.5]^3, let's use a grid of cells.
    
    # Let's try a 5x5x4 grid of BCC cells (5*5*4 = 100 cells, each with 2 points = 200 points)
    # Then select the best 100.
    
    # Actually, let's think differently. A BCC unit cell has 2 atoms.
    # If we use a grid of LxLxL unit cells, we get 2*L^3 atoms.
    # For 100 atoms, we need L such that 2*L^3 >= 100, so L=4 gives 128, L=3 gives 54.
    # Let's use L=4, giving 128 positions, then pick the best 100.
    
    L = 4
    # BCC lattice points in a LxLxL grid of unit cells
    # Corner points: (i, j, k) for i,j,k in 0..L
    # Center points: (i+0.5, j+0.5, k+0.5) for i,j,k in 0..L-1
    
    corners = []
    for i in range(L + 1):
        for j in range(L + 1):
            for k in range(L + 1):
                corners.append([i, j, k])
    
    centers = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                centers.append([i + 0.5, j + 0.5, k + 0.5])
    
    all_points = np.array(corners + centers, dtype=float)
    
    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans [0, L] x [0, L] x [0, L]
    # Scale factor: we want the range to be 1.0 (from -0.5 to 0.5)
    scale = 1.0 / L
    all_points = all_points * scale - 0.5
    
    # We have 125 + 64 = 189 points, need to select 100
    # Select the 100 points that are most "central" or just take the first 100
    # Better: select points to maximize minimum distance
    
    # For now, just take the first 100 as initial guess
    initial_points = all_points[:N].copy()
    
    # Ensure all points are strictly inside the cube
    eps = 1e-10
    initial_points = np.clip(initial_points, -0.5 + eps, 0.5 - eps)
    
    # Optimize: maximize the minimum pairwise distance and distance to walls
    # Use a repulsion-based approach: minimize negative of min distance
    # Or use a sum of inverse distances to encourage spreading
    
    def negative_min_distance(x):
        x = x.reshape(-1, 3)
        # Compute all pairwise distances
        diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = np.min(np.abs(x - 0.5), axis=1)  # distance to nearest wall
        min_wall_dist = np.min(wall_dists)
        
        # The radius is min(min_pair_dist/2, min_wall_dist)
        radius = min(min_pair_dist / 2.0, min_wall_dist)
        return -radius
    
    # Use L-BFGS-B or similar
    x0 = initial_points.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try multiple random restarts with different initial configurations
    best_x = x0
    best_score = -negative_min_distance(x0)
    
    for restart in range(20):
        if restart == 0:
            x_init = x0
        else:
            # Perturb the initial points
            x_init = x0 + np.random.randn(N * 3) * 0.01
            x_init = np.clip(x_init, -0.49, 0.49)
        
        result = minimize(
            negative_min_distance,
            x_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        score = -result.fun
        if score > best_score:
            best_score = score
            best_x = result.x
    
    # Final result
    result_points = best_x.reshape(N, 3)
    
    # Ensure validity: strictly inside cube
    result_points = np.clip(result_points, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return result_points
# EVOLVE-BLOCK-END
