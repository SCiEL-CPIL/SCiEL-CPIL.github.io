"""Program for packing 50 equal spheres in a unit cube.
Uses BCC-inspired initialization with repulsive potential energy minimization and local search polishing."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _repulsive_potential(flat_x, wall_margin=0.0):
    """Compute a repulsive potential energy that we want to MINIMIZE.
    
    We use an inverse-power repulsion between all pairs and walls.
    Minimizing this maximizes the minimum distance.
    """
    x = flat_x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, np.inf)
    
    # Pair repulsion: sum of 1/d^4 (stronger repulsion when close)
    # Using squared distances to avoid sqrt
    pair_energy = np.sum(1.0 / (dist_sq ** 2))
    
    # Wall distances: distance from each center to the nearest wall
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    
    # Wall repulsion: sum of 1/wall_dist^4
    # Clip to avoid infinity
    wall_dists_clipped = np.maximum(wall_dists, 1e-10)
    wall_energy = np.sum(1.0 / (wall_dists_clipped ** 4))
    
    return pair_energy + wall_energy


def _score(x):
    """Compute the actual score: minimum radius."""
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _bcc_initialization_54():
    """Initialize 54 centers on a BCC superlattice (3x3x3 cells)."""
    centers = []
    # BCC lattice in [-0.5, 0.5]^3
    # 3 cells per dimension, cell size a = 1/3
    a = 1.0 / 3.0
    offset = -0.5
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner-like point
                x = offset + i * a
                y = offset + j * a
                z = offset + k * a
                centers.append([x, y, z])
                
                # Body-center-like point
                x = offset + (i + 0.5) * a
                y = offset + (j + 0.5) * a
                z = offset + (k + 0.5) * a
                centers.append([x, y, z])
    
    return np.array(centers)


def _local_search_polish(x, steps=2000, initial_step_size=0.05):
    """Improve the packing using random local search moves."""
    x = x.copy()
    current_score = _score(x)
    step_size = initial_step_size
    
    for _ in range(steps):
        # Pick a random sphere
        idx = np.random.randint(N)
        
        # Generate a random move
        move = np.random.randn(3) * step_size
        new_x = x.copy()
        new_x[idx] += move
        
        # Keep inside cube (strictly)
        new_x[idx] = np.clip(new_x[idx], -0.4999, 0.4999)
        
        new_score = _score(new_x)
        
        if new_score > current_score:
            x = new_x
            current_score = new_score
            # Occasionally increase step size slightly if we found a big improvement?
            # Or just keep it.
        else:
            # Decrease step size if no improvement
            step_size *= 0.95
            if step_size < 1e-4:
                break
                
    return x, current_score


def construct():
    """Construct centers for 50 equal spheres in a unit cube."""
    # Generate 54 BCC points
    bcc_54 = _bcc_initialization_54()
    
    best_x = None
    best_score = 0.0
    
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    # Multiple restarts with different subsets of the 54 BCC points
    np.random.seed(42)
    
    for restart in range(10):
        # Randomly select 50 out of 54 points
        # This breaks symmetry and explores different local minima
        indices = np.random.choice(54, N, replace=False)
        x0 = bcc_54[indices]
        
        # Add small perturbation
        x0 += np.random.randn(N, 3) * 0.01
        x0 = np.clip(x0, -0.49, 0.49)
        
        # Gradient optimization
        result = minimize(
            _repulsive_potential,
            x0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt = np.clip(x_opt, -0.4999, 0.4999)
        
        # Local search polish
        x_polished, score_polished = _local_search_polish(x_opt, steps=1000)
        
        if score_polished > best_score:
            best_score = score_polished
            best_x = x_polished
            
    # Final verification: ensure validity
    if best_x is None:
        # Fallback to simple grid if something went wrong
        best_x = np.random.rand(N, 3) - 0.5
        best_x = np.clip(best_x, -0.499, 0.499)
    else:
        best_x = np.clip(best_x, -0.4999, 0.4999)
        
    return best_x
