"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_grad(centers, r, beta=10.0):
    """
    Compute the energy and gradient for a fixed radius r.
    Energy penalizes overlaps and wall violations using soft-plus.
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise interactions
    # Use vectorized calculation for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Violations: 2r - dist
    viol = 2.0 * r - dists
    
    # Soft-plus: (1/beta) * log(1 + exp(beta * x))
    # Gradient: sigma(beta * x)
    # To avoid overflow, use stable log-sum-exp or clipping
    # beta * viol can be large. 
    # sigma(x) = 1 / (1 + exp(-x))
    # For large positive x, sigma(x) ~ 1. For large negative, ~ 0.
    # Energy term: if x > 0, ~ x. If x < 0, ~ 0.
    # Actually, let's just use a quadratic penalty for simplicity and stability if beta is not critical.
    # But soft-plus is smoother.
    # Let's use a simple quadratic penalty for violations > 0. It's C1 continuous but not C2.
    # L-BFGS-B can handle it.
    # E = sum(max(0, 2r - d)^2)
    # Grad = -2 * max(0, 2r - d) * (c_i - c_j) / d
    
    mask = viol > 0
    if np.any(mask):
        # Energy contribution
        energy += np.sum(viol[mask]**2)
        
        # Gradient contribution
        # For each pair (i, j) where viol > 0:
        # dE/dc_i = -2 * viol_ij * (c_i - c_j) / d_ij
        # dE/dc_j = -2 * viol_ij * (c_j - c_i) / d_ij = 2 * viol_ij * (c_i - c_j) / d_ij
        
        # We need to accumulate gradients.
        # Vectorized accumulation is tricky with masks.
        # Loop over violating pairs if few, or use dense matrix multiplication.
        # Since N=100, 4950 pairs. A loop might be okay, but vectorized is better.
        
        # Let's compute the force vector F_ij = -2 * viol_ij * (c_i - c_j) / d_ij
        # We need to sum these forces for each i.
        # Note: d_ij is 0 on diagonal, but we masked inf.
        
        # Create a weight matrix W where W_ij = -2 * viol_ij / d_ij if viol > 0 else 0
        W = np.zeros_like(dists)
        W[mask] = -2.0 * viol[mask] / dists[mask]
        
        # Grad_i = sum_j W_ij * (c_i - c_j)
        # = sum_j W_ij * c_i - sum_j W_ij * c_j
        # = c_i * sum_j W_ij - (W @ c)_i
        
        sum_W = np.sum(W, axis=1)
        weighted_c = W @ centers
        
        grad += centers * sum_W[:, np.newaxis] - weighted_c
        
    # Wall interactions
    wall_dists = 0.5 - np.abs(centers)
    wall_viol = r - wall_dists
    
    wall_mask = wall_viol > 0
    if np.any(wall_mask):
        # Energy contribution
        energy += np.sum(wall_viol[wall_mask]**2)
        
        # Gradient contribution
        # dE/dc_ik = 2 * wall_viol_i * (-d wall_dist_i / dc_ik)
        # wall_dist_i = min_k (0.5 - |c_ik|)
        # The gradient is non-zero only for the active coordinate (the one achieving the min).
        # However, for smoothness, we can distribute or just pick the min.
        # Picking the min makes it non-smooth.
        # Alternative: Use sum of soft violations for all 6 walls?
        # E_wall = sum_i sum_k max(0, r - (0.5 - c_ik))^2 + max(0, r - (0.5 + c_ik))^2
        # This is always smooth (quadratic).
        # Let's switch to this definition for walls to ensure smooth gradients.
        
        # Recalculate wall energy and grad with this new definition
        # Reset wall part of energy and grad
        # We already added pairwise energy. We need to subtract the previous wall energy if we changed logic.
        # Let's restart the function logic to be consistent.
        
        pass # We will rewrite the function below.
        
    return energy, grad

def _energy_and_grad_v2(centers, r):
    """
    Smooth energy function using quadratic penalties for all constraints.
    Pairwise: max(0, 2r - d_ij)^2
    Wall: sum_k max(0, r - (0.5 - c_ik))^2 + max(0, r - (0.5 + c_ik))^2
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    
    viol = 2.0 * r - dists
    mask = viol > 0
    
    if np.any(mask):
        energy += np.sum(viol[mask]**2)
        
        # W_ij = -2 * viol_ij / d_ij
        W = np.zeros_like(dists)
        W[mask] = -2.0 * viol[mask] / dists[mask]
        
        sum_W = np.sum(W, axis=1)
        weighted_c = W @ centers
        grad += centers * sum_W[:, np.newaxis] - weighted_c
        
    # 2. Walls
    # For each center i and axis k:
    # Violation right: r - (0.5 - c_ik) = r - 0.5 + c_ik
    # Violation left: r - (0.5 + c_ik) = r - 0.5 - c_ik
    
    # Vectorized
    # c_ik ranges
    # Right wall dist: 0.5 - c_ik
    # Left wall dist: 0.5 + c_ik
    
    dist_right = 0.5 - centers
    dist_left = 0.5 + centers
    
    viol_right = r - dist_right
    viol_left = r - dist_left
    
    mask_right = viol_right > 0
    mask_left = viol_left > 0
    
    if np.any(mask_right):
        energy += np.sum(viol_right[mask_right]**2)
        # dE/dc_ik = 2 * viol_right * (1)  (since d(r - 0.5 + c)/dc = 1)
        grad[mask_right] += 2.0 * viol_right[mask_right]
        
    if np.any(mask_left):
        energy += np.sum(viol_left[mask_left]**2)
        # dE/dc_ik = 2 * viol_left * (-1) (since d(r - 0.5 - c)/dc = -1)
        grad[mask_left] -= 2.0 * viol_left[mask_left]
        
    return energy, grad

def _minimize_energy(centers, r, max_iter=200, time_limit=5.0):
    """Minimize energy for fixed r."""
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _energy_and_grad_v2(c, r)
        return e, g.flatten()
        
    try:
        res = minimize(
            obj, centers.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    centers = _grid_init(N)
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Iterative radius increase
    # Start with a small step, increase it if successful
    step = 0.001
    
    while time.time() < deadline - 10.0:
        # Try to increase radius
        new_r = current_r + step
        
        # Minimize energy for new_r
        optimized_centers = _minimize_energy(centers, new_r, max_iter=50, time_limit=2.0)
        
        # Check if the configuration is valid for new_r
        # Compute the actual radius achieved
        achieved_r = _compute_radius(optimized_centers)
        
        if achieved_r >= new_r - 1e-6:
            # Success
            centers = optimized_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            
            # Increase step slightly
            step *= 1.05
        else:
            # Failure
            # Reduce step
            step *= 0.5
            if step < 1e-6:
                break
            
            # Try again with smaller step? Or just retry with same r?
            # Let's retry with the same r but more iterations
            optimized_centers = _minimize_energy(centers, current_r, max_iter=100, time_limit=5.0)
            centers = optimized_centers
    
    # Final polish
    if time.time() < deadline - 5.0:
        # Use the best radius found and polish
        final_r = best_r
        polished_centers = _minimize_energy(best_centers, final_r, max_iter=200, time_limit=5.0)
        final_r_achieved = _compute_radius(polished_centers)
        if final_r_achieved > best_r:
            best_r = final_r_achieved
            best_centers = polished_centers
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
