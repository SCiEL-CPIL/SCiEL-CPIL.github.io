import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
CUBE_HALF = 0.5

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
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
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
    np.fill_diagonal(dists_sq, 1e-15)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
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
    wall_dists = 0.5 - max_abs
    
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

def _soft_min_obj_grad(flat_pos, beta=100.0):
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
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    # Handle zero sign
    signs[signs == 0] = 1.0
    
    # grad_pos[i, k] += -w_k * sign(pos_ik)
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    # Gradient of -soft_min is -grad(soft_min)
    return -soft_min_val, -grad_pos.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
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

def _minimize_soft_min(init_centers, beta, max_iter=300, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the negative soft-min objective.
    """
    def objective_and_grad(x):
        val, g = _soft_min_obj_grad(x, beta)
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
        return new_centers, res.success
    except Exception:
        return init_centers, False

def _generate_initial_positions():
    """
    Generate multiple initial positions using different strategies.
    """
    candidates = []
    
    # 1. FCC-inspired lattice
    # FCC: positions at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in unit cell
    # Scale to fit in [-0.5, 0.5]^3
    # Use a 5x5x1 grid of FCC cells
    fcc_points = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(1):
                base_x = (ix + 0.0) / 5.0 - 0.5
                base_y = (iy + 0.0) / 5.0 - 0.5
                base_z = (iz + 0.0) / 1.0 - 0.5
                # FCC offsets
                fcc_points.append([base_x, base_y, base_z])
                fcc_points.append([base_x + 0.5/5.0, base_y + 0.5/5.0, base_z])
                fcc_points.append([base_x + 0.5/5.0, base_y, base_z + 0.5/1.0])
                fcc_points.append([base_x, base_y + 0.5/5.0, base_z + 0.5/1.0])
    
    fcc_points = np.array(fcc_points[:N])
    # Scale to fit in cube
    fcc_points = fcc_points * 0.9
    candidates.append(fcc_points)
    
    # 2. Simple cubic grid 5x5x4
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 3. Simple cubic grid 4x5x5
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    candidates.append(grid[:N])
    
    # 4. Random with good spread
    for seed in [42, 123, 999, 777]:
        np.random.seed(seed)
        rand_pts = np.random.uniform(-0.45, 0.45, (N, 3))
        candidates.append(rand_pts)
    
    # 5. Jittered FCC
    np.random.seed(456)
    jitter = np.random.normal(0, 0.02, (N, 3))
    candidates.append(np.clip(candidates[0] + jitter, -0.5, 0.5))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Generate initial configurations
    candidates = _generate_initial_positions()
    
    for init_idx, init_centers in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
            
        current_centers = init_centers.copy()
        current_centers = np.clip(current_centers, -HALF, HALF)
        
        # Phase 1: Incremental radius targeting
        R_target = 0.05
        max_R = 0.13
        step_size = 0.002
        
        while R_target < max_R:
            if time.time() > deadline - 15.0:
                break
                
            new_centers, energy, success = _minimize_energy(
                current_centers, R_target, max_iter=300
            )
            
            if energy < 1e-6:
                current_centers = new_centers
                R_target += step_size
            else:
                # Try a smaller step
                if step_size > 0.0005:
                    step_size *= 0.5
                    R_target -= step_size
                    R_target = max(R_target, 0.03)
                else:
                    break
        
        # Phase 2: Soft-min polishing with increasing beta
        if time.time() < deadline - 10.0:
            for beta in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 8.0:
                    break
                
                polished, success = _minimize_soft_min(
                    current_centers, beta, max_iter=200
                )
                
                r_polished = _compute_radius(polished)
                r_current = _compute_radius(current_centers)
                
                if r_polished > r_current:
                    current_centers = polished
                    r_current = r_polished
        
        # Phase 3: Local perturbation search
        if time.time() < deadline - 5.0:
            np.random.seed(init_idx + 1000)
            for perturb_trial in range(3):
                if time.time() > deadline - 4.0:
                    break
                
                perturbation_scale = 0.01 / (perturb_trial + 1)
                perturbed = current_centers + np.random.normal(
                    0, perturbation_scale, (N, 3)
                )
                perturbed = np.clip(perturbed, -HALF, HALF)
                
                # Quick soft-min polish on perturbed
                polished, _ = _minimize_soft_min(
                    perturbed, 100.0, max_iter=100
                )
                
                r_perturbed = _compute_radius(polished)
                r_current = _compute_radius(current_centers)
                
                if r_perturbed > r_current:
                    current_centers = polished
                    r_current = r_perturbed
        
        # Update best
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()
    
    # Fallback
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
