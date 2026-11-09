"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the minimum radius for configuration c."""
    c = c.reshape(N, 3)
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    # Pairwise distances
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(np.min(wall_dists), min_pair_dist / 2.0)


def _soft_min_radius(c, alpha=100.0):
    """Smooth surrogate using log-sum-exp to approximate -min radius."""
    c = c.reshape(N, 3)
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    # Pairwise distances
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_radii = dists / 2.0
    
    # All constraints as negative values (we want to maximize the minimum)
    # For smooth min: -logsumexp(-x * alpha) / alpha
    all_vals = np.concatenate([wall_dists, pair_radii])
    # Clip to avoid numerical issues
    all_vals = np.maximum(all_vals, 1e-10)
    
    # Soft min approximation
    log_sum = np.log(np.sum(np.exp(-alpha * all_vals)))
    soft_min = -log_sum / alpha
    
    return soft_min


def _fcc_lattice_init():
    """Generate initial configuration based on FCC lattice structure."""
    # FCC lattice in a cube: place spheres at FCC positions
    # For better packing, we use a distorted FCC-like arrangement
    
    # Start with a simple grid and then apply FCC-like perturbations
    m = 5  # 5x5x5 grid gives 125 points, we need 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    
    # Generate all grid points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # For FCC-like structure, we can use a subset that approximates FCC
    # Actually, let's use a better approach: start with the grid and then
    # apply optimization to push toward better packing
    
    # Take first N points from grid
    x0 = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    x0 += np.random.randn(N, 3) * 0.01
    
    # Clip to bounds
    x0 = np.clip(x0, -0.49, 0.49)
    
    return x0


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Initial configuration using FCC-inspired lattice
    x0 = _fcc_lattice_init()
    
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    
    # Multiple optimization passes with increasing alpha for soft-min
    alphas = [50.0, 100.0, 200.0, 500.0]
    
    for alpha in alphas:
        if time.time() > deadline - 20:
            break
        
        # Smooth optimization pass
        try:
            result = minimize(
                lambda c: _soft_min_radius(c, alpha=alpha),
                best_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'maxls': 30, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            r_result = _compute_radius(result.x)
            if r_result > best_r + 1e-12:
                best_r = r_result
                best_x = result.x
        except Exception:
            pass
        
        # Exact objective refinement
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(
                lambda c: -_compute_radius(c),
                best_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            r_result = -result.fun
            if r_result > best_r + 1e-12:
                best_r = r_result
                best_x = result.x
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            x_start = np.clip(best_x, -0.49, 0.49)
            result = minimize(
                lambda c: -_compute_radius(c),
                x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            r_result = -result.fun
            if r_result > best_r + 1e-15:
                best_r = r_result
                best_x = result.x
        except Exception:
            pass
    
    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
