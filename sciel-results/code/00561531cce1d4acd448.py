"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5

# EVOLVE-BLOCK-START

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
    if np.any(centers < -HALF - 1e-12) or np.any(centers > HALF + 1e-12):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    wall_dists = HALF - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < -1e-12:
        return 0.0

    # Pairwise constraints
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
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq + 1e-15)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # 2. Wall terms (vectorized)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = HALF - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        # For each violating sphere, the gradient points away from the nearest wall
        # Find the axis with the maximum absolute coordinate
        max_k = np.argmax(abs_centers[i_valid_wall], axis=1)
        signs = np.sign(centers[i_valid_wall, max_k])
        signs[signs == 0] = 1.0
        grad_vals = 2.0 * violations_wall * signs
        
        # Scatter the gradient contributions
        np.add.at(grad, (i_valid_wall, max_k), grad_vals)
        
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=200, ftol=1e-12, gtol=1e-10, time_limit=10.0):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-HALF, HALF)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False, 'maxfun': 1000}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _soft_min_obj_grad(flat_pos, beta=50.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    This approximates the minimum as beta -> infinity.
    We minimize -soft_min.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # All effective distances (pair distances divided by 2, wall distances as-is)
    # Note: pair dist d corresponds to radius d/2. Wall dist d corresponds to radius d.
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -logsumexp(-beta * x) / beta
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. positions
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient
    # d(soft_min)/d(pos_i) = sum_j w_j * d(d_ij/2)/d(pos_i)
    # d(d_ij)/d(pos_i) = (pos_i - pos_j) / d_ij
    # So d(soft_min)/d(pos_i) = sum_j w_j * (pos_i - pos_j) / (2 * d_ij)
    
    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]
    
    # Vectorized accumulation
    i_idx = iu[0]
    j_idx = iu[1]
    d_vals = pair_dists
    
    # Avoid division by zero
    valid = d_vals > 1e-14
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = d_vals[valid]
        w_v = pair_weights[valid]
        
        vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
        coeffs = w_v / (2.0 * d_v) # shape (n_valid,)
        
        # grad_pos[i] += coeff * vec
        np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
        # grad_pos[j] -= coeff * vec
        np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    # d(soft_min)/d(pos_ik) = w_k * d(d_wall_ik)/d(pos_ik)
    # d_wall_ik = HALF - |pos_ik|
    # d(d_wall)/d(pos_ik) = -sign(pos_ik)
    # So d(soft_min)/d(pos_ik) = -w_k * sign(pos_ik)
    
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    # Handle zero sign
    signs[signs == 0] = 1.0
    
    # grad_pos[i, k] += -w_k * sign(pos_ik)
    # wall_weights is flattened (N*3)
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    # Gradient of -soft_min is -grad(soft_min)
    return -soft_min_val, -grad_pos.flatten()

def _minimize_soft_min(init_centers, beta, max_iter=300, ftol=1e-12, gtol=1e-10, time_limit=10.0):
    """
    Minimizes the negative soft-min objective.
    """
    def objective_and_grad(x):
        val, g = _soft_min_obj_grad(x, beta)
        return val, g

    bounds = [(-HALF, HALF)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False, 'maxfun': 1500}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers, res.success
    except Exception:
        return init_centers, False

def _generate_initial_positions():
    """
    Generate initial positions using a simple cubic grid with some jitter.
    """
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    grid[:N] += np.random.randn(N, 3) * 0.01
    grid[:N] = np.clip(grid[:N], -HALF + 0.01, HALF - 0.01)
    
    return grid[:N].copy()

def _local_optimization(centers, time_limit=5.0):
    """
    Do a local optimization pass to improve the packing using soft-min objective.
    """
    best_centers = centers.copy()
    best_score = _compute_radius(best_centers)
    
    start_time = time.time()
    
    # Try a few different beta values
    for beta in [30, 50, 100]:
        if time.time() > start_time + time_limit:
            break
            
        new_centers, success = _minimize_soft_min(best_centers, beta, max_iter=200, time_limit=time_limit)
        
        if success:
            new_score = _compute_radius(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
    
    return best_centers, best_score

def _simulated_annealing(centers, max_time=15.0):
    """Run simulated annealing to maximize the minimum radius."""
    best_centers = centers.copy()
    best_score = _compute_radius(best_centers)
    
    current_centers = centers.copy()
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = 0.005
    T_min = 1e-8
    cooling = 0.999
    
    np.random.seed(42)
    random.seed(42)
    
    step_size = 0.01
    
    while time.time() < deadline:
        # Check for time
        if time.time() > deadline - 2.0:
            break
        
        # Propose a move: randomly pick a few spheres and perturb them
        n_move = np.random.randint(1, 4)
        indices = np.random.choice(N, n_move, replace=False)
        
        old_positions = current_centers[indices].copy()
        
        # Perturb
        perturb = np.random.randn(n_move, 3) * step_size
        new_positions = old_positions + perturb
        
        # Keep within bounds
        new_positions = np.clip(new_positions, -HALF + 0.01, HALF - 0.01)
        
        current_centers[indices] = new_positions
        
        new_score = _compute_radius(current_centers)
        
        # Accept or reject
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
        else:
            # Reject
            current_centers[indices] = old_positions
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Adapt step size
        if new_score > current_score:
            step_size *= 1.01
        else:
            step_size *= 0.99
        step_size = np.clip(step_size, 0.001, 0.05)
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with grid
    centers = _generate_initial_positions()
    best_centers = centers.copy()
    best_score = _compute_radius(best_centers)
    
    # Run multiple optimization phases
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            current = centers.copy()
        else:
            # Add noise to previous best
            current = best_centers + np.random.randn(N, 3) * 0.05
            current = np.clip(current, -HALF + 0.01, HALF - 0.01)
        
        # Run simulated annealing
        if time.time() < deadline - 40.0:
            sa_centers, sa_score = _simulated_annealing(current, max_time=15.0)
            if sa_score > best_score:
                best_score = sa_score
                best_centers = sa_centers
        
        # Local optimization with soft-min
        if time.time() < deadline - 10.0:
            local_centers, local_score = _local_optimization(best_centers, time_limit=10.0)
            if local_score > best_score:
                best_score = local_score
                best_centers = local_centers
        
        # Try energy minimization with target radius
        if time.time() < deadline - 5.0:
            target_R = best_score * 1.01
            energy_centers, energy_val, success = _minimize_energy(best_centers, target_R, max_iter=100, time_limit=5.0)
            if success:
                energy_score = _compute_radius(energy_centers)
                if energy_score > best_score:
                    best_score = energy_score
                    best_centers = energy_centers
    
    # Final check
    if time.time() < deadline - 5.0:
        # One more local optimization pass
        final_centers, final_score = _local_optimization(best_centers, time_limit=5.0)
        if final_score > best_score:
            best_centers = final_centers
            best_score = final_score
    
    return best_centers

# EVOLVE-BLOCK-END
