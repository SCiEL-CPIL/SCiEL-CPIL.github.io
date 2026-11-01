import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _potential_energy(centers, r):
    """
    Compute the potential energy for a given radius r.
    Energy is sum of squared violations of distance constraints.
    Violation for pair (i,j): max(0, 2r - dist_ij)^2
    Violation for wall (i,k): max(0, r - (0.5 - |c_ik|))^2
    """
    c = centers
    energy = 0.0
    
    # Wall violations
    wall_dists = 0.5 - np.abs(c)
    wall_viol = np.maximum(0, r - wall_dists)
    energy += np.sum(wall_viol ** 2)
    
    # Pairwise violations
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = np.maximum(0, 2 * r - dists)
    # Only count i < j to avoid double counting, or just sum all and divide by 2?
    # Since we want to minimize total violation, summing all i!=j is fine, 
    # but let's be consistent. The gradient will handle the symmetry if we sum all.
    # Actually, summing i<j is more efficient.
    iu_i, iu_j = np.triu_indices(N, k=1)
    pair_viol_upper = pair_viol[iu_i, iu_j]
    energy += np.sum(pair_viol_upper ** 2)
    
    return energy

def _potential_energy_grad(centers, r):
    """
    Compute gradient of the potential energy with respect to centers.
    """
    c = centers
    grad = np.zeros_like(c)
    
    # Wall gradient
    wall_dists = 0.5 - np.abs(c)
    wall_viol = np.maximum(0, r - wall_dists)
    if np.any(wall_viol > 0):
        # d/dx max(0, r - (0.5 - |x|))^2 = 2 * max(0, ...) * d/dx(r - 0.5 + |x|)
        # d/dx |x| = sign(x)
        signs = np.sign(c)
        # If c is 0, sign is 0, but derivative of |x| at 0 is undefined. 
        # We can treat it as 0 or random. 0 is safer for stability.
        grad += -2 * wall_viol * signs
        
    # Pairwise gradient
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    
    pair_viol = np.maximum(0, 2 * r - dists)
    
    # d/dc_i sum_{j != i} max(0, 2r - d_ij)^2
    # = sum_{j != i} 2 * max(0, 2r - d_ij) * (-1) * d/dc_i (2r - d_ij)
    # d/dc_i (2r - d_ij) = - d/dc_i d_ij = - (c_i - c_j) / d_ij
    # So term is 2 * viol_ij * (c_i - c_j) / d_ij
    
    # We need to sum over j for each i.
    # term_ij = 2 * pair_viol[i,j] * (c_i - c_j) / dists[i,j]
    # grad[i] = sum_{j != i} term_ij
    
    # To do this efficiently:
    # coeff[i,j] = 2 * pair_viol[i,j] / dists[i,j]
    # grad[i] = sum_j coeff[i,j] * (c_i - c_j)
    #          = c_i * sum_j coeff[i,j] - sum_j coeff[i,j] * c_j
    
    coeff = np.zeros_like(dists)
    coeff[mask] = 2 * pair_viol[mask] * inv_dists[mask]
    
    # Sum of coefficients for each i
    sum_coeff = np.sum(coeff, axis=1)
    
    # Sum of coeff[i,j] * c_j
    # coeff is N x N. c is N x 3.
    # weighted_c[j] = sum_i coeff[i,j] * c[i]? No, we want sum_j coeff[i,j] c[j].
    # This is coeff.T @ c.
    weighted_c = coeff.T @ c
    
    grad += sum_coeff[:, np.newaxis] * c - weighted_c
    
    return grad

def _force_directed_refine(centers, r, deadline, max_iters=2000, lr=0.01):
    """
    Refine centers to satisfy constraints for radius r using gradient descent.
    Returns (centers, success) where success is True if energy is low enough.
    """
    c = centers.copy()
    current_energy = _potential_energy(c, r)
    best_energy = current_energy
    best_c = c.copy()
    
    # Adaptive learning rate
    lr_current = lr
    patience = 50
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        grad = _potential_energy_grad(c, r)
        
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # SGD step
        c_new = c - lr_current * grad
        c_new = np.clip(c_new, -0.5, 0.5)
        
        new_energy = _potential_energy(c_new, r)
        
        if new_energy < current_energy:
            c = c_new
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_c = c.copy()
        else:
            no_improve += 1
            # Reduce learning rate if stuck
            if no_improve > patience:
                lr_current *= 0.5
                no_improve = 0
                if lr_current < 1e-6:
                    break
        
        # Early stopping if energy is effectively zero
        if current_energy < 1e-10:
            break
            
    return best_c, best_energy < 1e-8

def _init_grid_perturbed(rng, nx, ny, nz, noise_scale=0.01):
    """Perturbed grid initialization."""
    xs = np.linspace(-0.5 + 0.03, 0.5 - 0.03, nx)
    ys = np.linspace(-0.5 + 0.03, 0.5 - 0.03, ny)
    zs = np.linspace(-0.5 + 0.03, 0.5 - 0.03, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_fcc_like(rng):
    """
    Initialize with FCC-like lattice points.
    """
    # Generate a dense grid and select points close to FCC positions
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # We'll just use a perturbed grid as a proxy, or generate specific FCC sites.
    # For 100 points, a 5x5x4 grid is standard.
    # Let's try to align with FCC by shifting alternating layers.
    nx, ny, nz = 5, 5, 4
    centers = np.zeros((N, 3))
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                centers[idx] = [x, y, z]
                idx += 1
    centers += rng.normal(0, 0.005, centers.shape)
    return np.clip(centers, -0.499, 0.499)

def _init_random(rng):
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _find_max_radius(centers, deadline, r_low=0.1, r_high=0.15):
    """
    Binary search for the maximum radius r such that a valid configuration exists
    starting from `centers`.
    """
    best_c = centers.copy()
    best_r = r_low
    
    # First, ensure we have a valid configuration for r_low
    # If not, we might need to expand them first? 
    # The prompt implies we start with some centers. 
    # Let's assume the initial centers are "reasonable".
    # If r_low is too high for the initial centers, the force-directed method will try to fix it.
    # If it fails, we should lower r_low.
    
    # Let's do a few iterations of binary search
    for _ in range(10):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a valid configuration for r_mid
        # We use the current best_c as the starting point for the force-directed method
        c_test, success = _force_directed_refine(best_c, r_mid, deadline - 5.0, max_iters=500)
        
        if success:
            r_low = r_mid
            best_c = c_test
            best_r = r_mid
        else:
            r_high = r_mid
            
        if r_high - r_low < 1e-5:
            break
            
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_init_grid_perturbed(rng, 5, 5, 4, 0.01))
    inits.append(_init_grid_perturbed(rng, 4, 5, 5, 0.01))
    inits.append(_init_grid_perturbed(rng, 5, 4, 5, 0.01))
    inits.append(_init_fcc_like(rng))
    inits.append(_init_random(rng))
    
    # Known best radius is ~0.11138
    # We will search in range [0.10, 0.12]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        c = init_centers.copy()
        
        # Try to find a good radius and configuration
        # We start with a moderate radius and try to increase it
        # Or we can use the binary search on radius directly.
        
        # Let's try to optimize the radius directly using the binary search helper
        # But we need a good starting point for the force-directed method.
        # The force-directed method is robust, so we can start from the init.
        
        # We will run the binary search for a few seconds
        # The deadline for this init is deadline - 20.0 (leaving time for others)
        # But we should distribute time.
        time_per_init = (deadline - start_time - 10.0) / len(inits)
        init_deadline = start_time + (init_idx + 1) * time_per_init + 10.0 # rough allocation
        
        # Actually, let's just use a global deadline and break if we exceed it.
        # But we want to try all inits if possible.
        
        # Let's allocate 20 seconds per init, max 5 inits -> 100s.
        # Remaining 45s for final polish.
        
        c_opt, r_opt = _find_max_radius(c, deadline - 40.0, r_low=0.10, r_high=0.12)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
            
        # If time allows, do a final polish on the best center found so far
        if time.time() < deadline - 10.0:
             # Try to push the radius up a bit more from the current best
             # Use a very high precision binary search step
             pass # _find_max_radius already does this
            
    # Final polish
    if best_centers is not None and time.time() < deadline - 5.0:
        # Try to increase radius slightly
        # We can run a final force-directed refinement with a slightly higher r
        # and see if it converges.
        
        # Let's try a few steps of increasing r
        current_r = best_radius
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            next_r = current_r + 0.0005
            c_test, success = _force_directed_refine(best_centers, next_r, deadline - 2.0, max_iters=200)
            if success:
                best_centers = c_test
                best_radius = next_r
            else:
                break

    # Fallback
    if best_centers is None:
        best_centers = _init_grid_perturbed(rng, 5, 5, 4, 0.01)
        best_radius = _compute_radius(best_centers)
        
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
