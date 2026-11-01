"""Packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 100
HALF = 0.5  # half side length


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    # Distance to walls
    wall_dists = np.min(HALF - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(0.5)  # only pairs closer than 0.5 matter for radius < 0.25
    if len(pairs) == 0:
        min_pair = 2.0  # no close pairs, limited by walls
    else:
        min_pair = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
        min_pair /= 2.0
    return min(min_wall, min_pair)


def _fcc_init(n_spheres):
    """Generate an FCC-like initial configuration."""
    # FCC lattice in [0,1]^3
    a = 1.0  # cube side
    # FCC primitive vectors
    # We'll place points on an FCC lattice and take the first n_spheres
    # FCC has 4 atoms per unit cell
    # For 100 spheres, we need about 25 unit cells
    
    # Alternative: use a body-centered cubic or just random with some structure
    # Let's use a simple approach: place on a grid with some jitter
    
    # Actually, let's try a more structured approach:
    # Use a simple cubic grid with 5x5x5 = 125 points, take first 100
    # Then optimize
    
    m = 5
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                pts.append([(i + 0.5) / m - 0.5, 
                           (j + 0.5) / m - 0.5, 
                           (k + 0.5) / m - 0.5])
    pts = np.array(pts[:n_spheres])
    
    # Add small random jitter to break symmetry
    pts += np.random.randn(*pts.shape) * 0.01
    # Clip to stay inside
    pts = np.clip(pts, -HALF + 0.001, HALF - 0.001)
    return pts


def _anneal(centers, time_limit, n_restarts=3):
    """Run simulated annealing to optimize the packing."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    t0 = time.time()
    
    for restart in range(n_restarts):
        # For subsequent restarts, add noise to best solution
        if restart > 0:
            centers = best_centers + np.random.randn(*centers.shape) * 0.05
            centers = np.clip(centers, -HALF + 0.001, HALF - 0.001)
        
        current_radius = _compute_radius(centers)
        
        # Annealing parameters
        T_start = 0.02
        T_end = 0.0001
        n_iter = 5000
        
        for it in range(n_iter):
            # Check time
            if time.time() - t0 > time_limit - 5:
                break
            
            # Choose a random sphere to move
            i = np.random.randint(N)
            old_pos = centers[i].copy()
            
            # Perturbation: random step
            step_size = T_start * 0.1
            centers[i] += np.random.randn(3) * step_size
            
            # Keep inside cube
            centers[i] = np.clip(centers[i], -HALF + 0.0001, HALF - 0.0001)
            
            new_radius = _compute_radius(centers)
            
            # Metropolis criterion
            if new_radius >= current_radius:
                current_radius = new_radius
            else:
                T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
                delta = new_radius - current_radius
                if np.random.random() < np.exp(delta / max(T, 1e-10)):
                    current_radius = new_radius
                else:
                    centers[i] = old_pos
            
            # Update best
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
            
            # Adaptive step size
            if it % 500 == 499:
                # Estimate current scale
                pass  # Keep it simple
    
    return best_centers, best_radius


def construct():
    """Construct the centers of 100 equal spheres in the unit cube."""
    t0 = time.time()
    time_limit = 115  # safety margin
    
    # Initial configuration
    centers = _fcc_init(N)
    
    # Run annealing
    centers, radius = _anneal(centers, time_limit)
    
    # Final check: ensure all centers are strictly inside
    centers = np.clip(centers, -HALF + 1e-6, HALF - 1e-6)
    
    return centers
