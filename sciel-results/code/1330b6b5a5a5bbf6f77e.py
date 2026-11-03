import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    # r_i = min_k (0.5 - |c_{ik}|)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # r_{ij} = ||c_i - c_j|| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    E = sum_{i<j, d_ij < 2R} (2R - d_ij)^2 + sum_{i, w_i < R} (R - w_i)^2
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15) # +1e-15 for numerical stability
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs closer than 2R
    # To be efficient, we can iterate over upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Mask for pairs closer than 2R
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        # Energy contribution: sum(v^2)
        energy += np.sum(v_ij**2)
        
        # Gradient:
        # d/dx_i (2R - ||x_i - x_j||)^2 = 2(2R - d) * (-1) * (x_i - x_j)/d
        # = -2(2R - d)/d * (x_i - x_j)
        # d/dx_j ... = 2(2R - d)/d * (x_i - x_j)
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        # grad_i += -forces
        # grad_j += +forces
        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    # w_i = 0.5 - max_k |c_{ik}|
    # Violation if w_i < R, i.e., max_k |c_{ik}| > 0.5 - R
    # Let M_i = max_k |c_{ik}|
    # Violation amount v_i = M_i - (0.5 - R) if M_i > 0.5 - R
    
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    max_abs_idx = np.argmax(abs_centers, axis=1)
    
    # Threshold for violation
    threshold = 0.5 - R
    mask_wall = max_abs > threshold
    
    if np.any(mask_wall):
        v_wall = max_abs[mask_wall] - threshold
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(v_wall**2)
        
        # Gradient:
        # E = sum (M_i - T)^2
        # dE/dx_{i,k} = 2(M_i - T) * dM_i/dx_{i,k}
        # M_i = |x_{i, k*}| where k* is the index of the max abs component.
        # dM_i/dx_{i,k} = sign(x_{i,k}) if k == k*, else 0.
        
        for idx in i_valid_wall:
            v = v_wall[np.where(i_valid_wall == idx)[0][0]]
            k_star = max_abs_idx[idx]
            s = np.sign(centers[idx, k_star])
            if s == 0:
                s = 1.0 # Arbitrary, gradient is 0 at 0 anyway technically, but avoid div by 0 issues in logic if any
                
            grad[idx, k_star] += 2.0 * v * s

    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    Returns (new_centers, final_energy, success_flag)
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Use 140s for logic, 10s buffer.
    deadline = start_time + 140.0
    
    # 1. Initialization: 5x4x5 Grid
    # 5*4*5 = 100 points exactly.
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    # Add small noise to break symmetry slightly, help with convergence
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial optimization to get a good baseline
    # Start with a conservative radius
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
        # Fallback to random if grid failed
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.0005
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        # Try multiple starting points:
        # 1. Current best
        # 2. Perturbed current best (small)
        # 3. Perturbed current best (large)
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.005, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.02, size=(N, 3)))
        
        success = False
        best_candidate_pos = None
        best_candidate_r = current_r
        
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 300 if current_r < 0.11 else 500
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            # Check if the energy is low enough to consider the constraint satisfied
            # Energy < 1e-10 implies violations are very small
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-6:
                    # Success: we found a valid configuration for target_r
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not perfectly zero, if we improved the radius, we can take it
                # This allows gradual improvement even if target_r wasn't fully met
                actual_r = _compute_radius(new_pos)
                if actual_r > best_candidate_r:
                    best_candidate_r = actual_r
                    best_candidate_pos = new_pos
        
        if not success:
            consecutive_failures += 1
            
            # If we didn't reach the target, but found a better configuration than current best,
            # update the best position but keep current_r (or update to best_candidate_r if it's higher)
            if best_candidate_pos is not None and best_candidate_r > best_r:
                best_pos = best_candidate_pos
                best_r = best_candidate_r
                current_r = best_r
                
            # If we failed to increase radius, try to polish the current best position
            # to ensure it's locally optimal for its current radius
            if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=200)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        # Try to push slightly beyond current best with a very small step
        # and high precision
        target_r = best_r + 1e-5
        new_pos, energy, _ = _minimize_energy(best_pos, target_r, max_iter=1000, ftol=1e-16, gtol=1e-14)
        final_r = _compute_radius(new_pos)
        if final_r > best_r:
            best_pos = new_pos
            best_r = final_r

    return best_pos

# EVOLVE-BLOCK-END
