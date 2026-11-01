"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    n = len(centers)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For N=100, we can compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _soft_min_penalty(centers, beta=100.0):
    """
    Smooth surrogate for the hard-sphere constraint using log-sum-exp.
    We want to maximize min of all constraints (pairwise distances/2 and wall distances).
    This is equivalent to minimizing -min, which is hard.
    
    Instead, we minimize a potential energy that penalizes close pairs and wall violations.
    Using a soft-min approach: we want all distances to be >= 2r.
    
    Let's define the objective as minimizing a function that is large when spheres are close
    or when they are close to walls.
    
    Actually, a better formulation: minimize the maximum "violation" or use a potential.
    
    Let's use the following energy:
    E = sum_{i<j} (1/d_{ij})^p + sum_i (1/w_i)^p
    where d_{ij} is distance between i and j, w_i is distance to nearest wall.
    This is large when spheres are close or near walls. Minimizing this pushes them apart.
    """
    n = len(centers)
    p = 8.0  # High power to approximate hard constraint
    
    # Pairwise
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    pair_energy = np.sum(dists_sq ** (-p/2.0))
    
    # Wall
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-12)
    wall_energy = np.sum(wall_dists ** (-p))
    
    return pair_energy + wall_energy


def _soft_min_penalty_grad(centers):
    """Gradient of the soft-min penalty."""
    n = len(centers)
    p = 8.0
    p_half = p / 2.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # d/dx_i sum_{j != i} (d_{ij}^2)^{-p/2}
    # = sum_{j != i} (-p/2) (d_{ij}^2)^{-p/2 - 1} * 2 * (x_i - x_j)
    # = -p * sum_{j != i} (d_{ij}^2)^{-(p+2)/2} * (x_i - x_j)
    
    coeff = -p * dists_sq ** (-(p + 2) / 2.0)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diffs, axis=1)
    
    # Wall gradient
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-12)
    
    # d/dx_i (w_i)^{-p} = -p * (w_i)^{-p-1} * dw_i/dx_i
    # w_i = 0.5 - |x_i|
    # dw_i/dx_i = -sign(x_i)
    # So d/dx_i = -p * (w_i)^{-p-1} * (-sign(x_i)) = p * (w_i)^{-p-1} * sign(x_i)
    
    signs = np.sign(centers)
    signs[np.abs(centers) < 1e-12] = 1.0  # Arbitrary sign at origin
    
    wall_grad = p * (wall_dists ** (-p - 1.0)) * signs
    
    return grad + wall_grad


def _objective_and_grad(centers_flat):
    """Objective and gradient for scipy minimize."""
    centers = centers_flat.reshape(N, 3)
    val = _soft_min_penalty(centers)
    grad = _soft_min_penalty_grad(centers).reshape(-1)
    return val, grad


def _make_grid_init(m=None):
    """Create a grid initialization."""
    if m is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Start with grid initialization
    centers = _make_grid_init()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Run optimization with different random perturbations
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the initial grid
        if restart == 0:
            x0 = centers.copy()
        else:
            x0 = centers + np.random.randn(N, 3) * 0.01
            # Clip to stay inside cube
            x0 = np.clip(x0, -0.49, 0.49)
        
        try:
            result = minimize(
                _objective_and_grad,
                x0.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centers = result.x.reshape(N, 3)
            # Ensure centers are inside the cube
            new_centers = np.clip(new_centers, -0.5 + 1e-10, 0.5 - 1e-10)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_and_grad,
                best_centers.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-10, 0.5 - 1e-10)
            
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Final validation
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
