"""Program for packing 50 equal spheres in a unit cube.
Uses BCC-inspired initialization with repulsive potential energy minimization."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _repulsive_potential(flat_x):
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
    dist = np.sqrt(dist_sq)
    
    # Pair repulsion: sum of 1/d^4 (stronger repulsion when close, smoother gradient)
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
    wall_dists_clipped = np.maximum(wall_dists, 1e-8)
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


def _bcc_initialization(seed=42):
    """Initialize 50 centers on a distorted BCC superlattice.
    
    BCC lattice: For a cube of side L, BCC has 2 atoms per conventional cell.
    We create a 3x3x3 grid of BCC unit cells (27 cells * 2 atoms = 54),
    then randomly select 50 of them with slight perturbation.
    """
    np.random.seed(seed)
    
    n_cells = 3  # 3x3x3 = 27 BCC cells
    cell_size = 1.0 / n_cells
    
    centers = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # Base position of this sub-cube
                cx = -0.5 + (i + 0.5) * cell_size
                cy = -0.5 + (j + 0.5) * cell_size
                cz = -0.5 + (k + 0.5) * cell_size
                
                # BCC: two atoms - one at corner offset, one at body center
                # Corner-like atom (offset toward one corner)
                offset = 0.25 * cell_size
                centers.append([cx - offset, cy - offset, cz - offset])
                # Body center atom
                centers.append([cx + offset, cy + offset, cz + offset])
    
    centers = np.array(centers)
    
    # Shuffle and pick 50 to allow for different subsets
    indices = np.random.permutation(54)
    centers = centers[indices[:N]]
    
    # Add small random perturbation to break symmetry
    centers += np.random.randn(N, 3) * 0.01
    
    # Clip to stay within cube with margin
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _soft_min_potential(flat_x, beta):
    """Compute a smoothed negative minimum distance potential.
    
    Minimizing this function maximizes the minimum distance (pair or wall).
    Uses log-sum-exp for numerical stability.
    """
    x = flat_x.reshape(N, 3)
    
    # Pair distances (radius contribution is d/2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq) / 2.0
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    
    # Extract unique pair distances
    triu_indices = np.triu_indices(N, k=1)
    pair_dists = dist[triu_indices]
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Stable log-sum-exp calculation for -1/beta * log(sum(exp(-beta * d)))
    min_d = np.min(all_dists)
    # If all distances are very large, min_d is large.
    # We want to maximize min_d.
    # The function is -soft_min.
    
    # To avoid underflow in exp(-beta * (d - min_d)), note that (d - min_d) >= 0.
    # So -beta * (d - min_d) <= 0. exp is in (0, 1].
    shifted = -beta * (all_dists - min_d)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_d + log_sum / beta
    
    return -soft_min


def construct():
    """Construct centers for 50 equal spheres in a unit cube."""
    best_x = None
    best_score = 0.0
    
    # Try multiple BCC initializations with different seeds
    seeds = [42, 123, 456, 789, 1011, 2023, 3030, 4040]
    
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    for seed in seeds:
        x0 = _bcc_initialization(seed=seed)
        x0_flat = x0.flatten()
        
        # Stage 1: Optimize repulsive potential to spread points
        result = minimize(
            _repulsive_potential,
            x0_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt = np.clip(x_opt, -0.499, 0.499)
        
        # Stage 2: Refine using soft-min potential to maximize minimum distance
        # Increase beta to sharpen the minimum
        current_x = x_opt
        for beta in [10, 50, 100, 500]:
            def objective(flat_x):
                return _soft_min_potential(flat_x, beta)
            
            res = minimize(
                objective,
                current_x.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
            )
            current_x = res.x.reshape(N, 3)
            current_x = np.clip(current_x, -0.499, 0.499)
        
        x_refined = current_x
        s = _score(x_refined)
        if s > best_score:
            best_score = s
            best_x = x_refined
    
    # Final verification: ensure validity
    best_x = np.clip(best_x, -0.499, 0.499)
    
    return best_x
