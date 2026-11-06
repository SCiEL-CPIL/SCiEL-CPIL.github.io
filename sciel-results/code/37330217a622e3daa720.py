"""Program for packing 50 equal spheres in a unit cube.
Uses BCC-inspired initialization with repulsive potential energy minimization."""
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
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, np.inf)
    
    # Pair repulsion: sum of 1/d^2 (strong repulsion when close)
    pair_energy = np.sum(1.0 / (dist ** 2))
    
    # Wall distances: distance from each center to the nearest wall
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    
    # Wall repulsion: sum of 1/wall_dist^2
    # Clip to avoid infinity
    wall_dists_clipped = np.maximum(wall_dists, 1e-10)
    wall_energy = np.sum(1.0 / (wall_dists_clipped ** 2))
    
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


def _bcc_initialization():
    """Initialize 50 centers on a distorted BCC superlattice."""
    # BCC lattice: corners + body center of each cube
    # In a unit cube [-0.5, 0.5]^3, we'll create a BCC-like arrangement
    
    # Use a 3x3x3 grid of BCC unit cells (27 cells * 2 atoms = 54, take 50)
    # Or use a more compact arrangement
    
    # Let's try: 4x4x4 simple cubic = 64 points, take 50
    # But BCC is denser. Let's think differently.
    
    # BCC in a cube: for a cube of side L, BCC has 2 atoms per conventional cell.
    # For 50 spheres, we want roughly 25 BCC unit cells.
    # 3x3x3 = 27 cells, 54 atoms. We'll use 3x3x3 BCC and drop 4.
    
    n_cells = 3  # 3x3x3 = 27 BCC cells
    cell_size = 1.0 / n_cells
    
    centers = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # Corner of this sub-cube
                cx = -0.5 + (i + 0.0) * cell_size
                cy = -0.5 + (j + 0.0) * cell_size
                cz = -0.5 + (k + 0.0) * cell_size
                
                # BCC: corner atom and body-center atom
                # Corner atom
                centers.append([cx + 0.1 * cell_size, cy + 0.1 * cell_size, cz + 0.1 * cell_size])
                # Body center atom
                centers.append([cx + 0.5 * cell_size, cy + 0.5 * cell_size, cz + 0.5 * cell_size])
    
    centers = np.array(centers[:N])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.randn(N, 3) * 0.01
    
    # Clip to stay within cube
    centers = np.clip(centers, -0.45, 0.45)
    
    return centers


def construct():
    """Construct centers for 50 equal spheres in a unit cube."""
    # Initialize with BCC-like arrangement
    x0 = _bcc_initialization()
    
    # Flatten for optimization
    x0_flat = x0.flatten()
    
    # Define bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Optimize using L-BFGS-B
    result = minimize(
        _repulsive_potential,
        x0_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    x_opt = result.x.reshape(N, 3)
    
    # Ensure all points are strictly inside the cube
    x_opt = np.clip(x_opt, -0.499, 0.499)
    
    # Try multiple restarts with different initializations to find better local minima
    best_x = x_opt
    best_score = _score(x_opt)
    
    # A few more random restarts
    for restart in range(3):
        np.random.seed(100 + restart)
        x_restart = _bcc_initialization()
        x_restart += np.random.randn(N, 3) * (0.05 * (restart + 1))
        x_restart = np.clip(x_restart, -0.45, 0.45)
        
        res = minimize(
            _repulsive_potential,
            x_restart.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        x_try = res.x.reshape(N, 3)
        x_try = np.clip(x_try, -0.499, 0.499)
        
        s = _score(x_try)
        if s > best_score:
            best_score = s
            best_x = x_try
    
    # Final verification: ensure validity
    best_x = np.clip(best_x, -0.499, 0.499)
    
    return best_x
