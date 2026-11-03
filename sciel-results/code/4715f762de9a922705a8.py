"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _radius_for_centres(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    if n < 2:
        return 0.5 - np.max(np.abs(centres))
    # Pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def _score(centres):
    """Score function: the radius."""
    return _radius_for_centres(centres)

def _optimize_from_start(start, max_iter=2000, tol=1e-10):
    """
    Optimize from a starting configuration using a custom gradient ascent 
    on the minimum constraint.
    """
    centres = start.copy()
    n = len(centres)
    
    best_r = _radius_for_centres(centres)
    best_centres = centres.copy()
    
    for it in range(max_iter):
        # Compute all pairwise distances
        diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centres)
        
        # Current radius
        min_pair_dist = np.min(dists)
        min_wall = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # Check convergence
        if abs(current_r - best_r) < tol and it > 10:
            break
        
        # Compute subgradient
        grad = np.zeros_like(centres)
        
        # Determine which constraint is active
        if min_wall < min_pair_dist / 2.0:
            # Wall constraint active
            active_idx = np.argmin(wall_dists)
            i, k = divmod(active_idx, 3)
            # Gradient of (0.5 - |c_ik|) w.r.t. c_ik is -sign(c_ik)
            # We want to maximize min, so we move in direction of increasing the minimum
            # If c_ik > 0, wall is at 0.5, distance is 0.5 - c_ik, gradient is -1
            # If c_ik < 0, wall is at -0.5, distance is 0.5 + c_ik, gradient is +1
            # To increase distance, move away from wall: if c_ik > 0, decrease c_ik (move left)
            # Actually, let's think: we want to maximize 0.5 - |c_ik|
            # d(0.5 - |c|)/dc = -sign(c)
            # So gradient of the objective w.r.t. c is -sign(c)
            # But we're doing ascent on the min, so we add subgradient
            grad[i, k] = -np.sign(centres[i, k])
            if centres[i, k] == 0:
                grad[i, k] = 1.0  # arbitrary
        else:
            # Pair constraint active
            # Find the pair with minimum distance
            min_pair_idx = np.argmin(dists)
            i, j = divmod(min_pair_idx, n)
            if i == j:
                continue
            # Distance between i and j
            diff = centres[i] - centres[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Degenerate, random direction
                grad[i] = np.random.randn(3)
                grad[j] = -np.random.randn(3)
            else:
                # Gradient of |c_i - c_j| w.r.t. c_i is (c_i - c_j)/|c_i - c_j|
                # We want to maximize min distance, so we move i away from j and j away from i
                unit = diff / dist
                grad[i] = unit
                grad[j] = -unit
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Adaptive step size
        step = 0.01 * (best_r + 1e-6)
        if it < 50:
            step *= 0.5
        elif it < 200:
            step *= 0.8
        
        # Take step
        new_centres = centres + step * grad
        
        # Project back to valid region (keep inside cube with margin)
        new_centres = np.clip(new_centres, -0.49, 0.49)
        
        # Check if new radius is better
        new_r = _radius_for_centres(new_centres)
        if new_r > current_r - 1e-8:
            centres = new_centres
        else:
            # Reduce step
            step *= 0.5
            if step < 1e-12:
                break
            new_centres = centres + step * grad
            new_centres = np.clip(new_centres, -0.49, 0.49)
            centres = new_centres
    
    return best_centres, best_r

def _initial_grid(n_spheres):
    """Create an initial grid configuration."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres]

def _random_perturb(centres, sigma=0.02):
    """Randomly perturb centres."""
    perturbed = centres + np.random.randn(*centres.shape) * sigma
    return np.clip(perturbed, -0.49, 0.49)

def construct():
    """
    Construct 50 sphere centres in the unit cube [-1/2, 1/2]^3 to maximize
    the common radius.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with grid
    best_centres = _initial_grid(N)
    best_r = _score(best_centres)
    
    # Multiple random restarts with perturbation
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate perturbed start
        if restart == 0:
            start = best_centres.copy()
        else:
            start = _random_perturb(best_centres, sigma=0.05 * (1 + restart / 10))
        
        # Optimize
        opt_centres, opt_r = _optimize_from_start(start, max_iter=1000, tol=1e-10)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
        
        # If we found a good solution, try smaller perturbations
        if best_r > 0.13:
            # Do a few more fine-tuning restarts
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                fine_start = _random_perturb(best_centres, sigma=0.01)
                opt_centres, opt_r = _optimize_from_start(fine_start, max_iter=500, tol=1e-11)
                if opt_r > best_r:
                    best_r = opt_r
                    best_centres = opt_centres
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        final_centres, final_r = _optimize_from_start(best_centres, max_iter=5000, tol=1e-12)
        if final_r > best_r:
            best_r = final_r
            best_centres = final_centres
    
    # Ensure validity: clip to cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
