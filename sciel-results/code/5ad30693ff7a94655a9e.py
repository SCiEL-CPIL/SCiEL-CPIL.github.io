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
    dists = np.sqrt(dists_sq + 1e-15)
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
    # wall_weights is flattened (N*3)
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
    Generate initial positions using an FCC lattice structure.
    """
    a = 0.35
    points = []
    for ix in range(-2, 4):
        for iy in range(-2, 4):
            for iz in range(-2, 4):
                base_x = ix * a
                base_y = iy * a
                base_z = iz * a
                
                offsets = [
                    (0, 0, 0),
                    (0.5*a, 0.5*a, 0),
                    (0.5*a, 0, 0.5*a),
                    (0, 0.5*a, 0.5*a)
                ]
                
                for ox, oy, oz in offsets:
                    x = base_x + ox
                    y = base_y + oy
                    z = base_z + oz
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    if len(points) < N:
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        points = grid[:N].copy()
    elif len(points) > N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=N, replace=False)
        points = points[idx]
    
    rng = np.random.default_rng(42)
    pos = points + rng.normal(0, 0.005, size=(N, 3))
    pos = np.clip(pos, -0.49, 0.49)
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    best_pos = _generate_initial_positions()
    
    # Initial soft-min optimization to spread out spheres evenly
    # Use a low beta first for global exploration
    try:
        soft_pos, _ = _minimize_soft_min(best_pos, beta=20.0, max_iter=200)
        soft_r = _compute_radius(soft_pos)
        if soft_r > _compute_radius(best_pos):
            best_pos = soft_pos
    except Exception:
        pass
    
    # Initial optimization with a conservative radius using energy minimization
    initial_r = _compute_radius(best_pos)
    if initial_r < 0.08:
        initial_r = 0.08
    else:
        initial_r = initial_r * 0.95
        
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=300)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
        rng = np.random.default_rng(123)
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
    
    # 2. Iterative Radius Increase with adaptive step size
    current_r = best_r
    step_size = 0.0015
    
    consecutive_failures = 0
    max_failures = 4
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 8.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-9:
                break
        
        # Try multiple perturbations to escape local minima
        candidates = [best_pos.copy()]
        for scale in [0.002, 0.008, 0.02, 0.05]:
            perturbed = best_pos + rng_search.normal(0, scale, size=(N, 3))
            perturbed = np.clip(perturbed, -0.495, 0.495)
            candidates.append(perturbed)
        
        success = False
        best_candidate_pos = None
        best_candidate_r = current_r
        
        for start_pos in candidates:
            max_iter = 500 if current_r < 0.105 else 700
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-7:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                actual_r = _compute_radius(new_pos)
                if actual_r > best_candidate_r:
                    best_candidate_r = actual_r
                    best_candidate_pos = new_pos.copy()
        
        if not success:
            if best_candidate_pos is not None and best_candidate_r > current_r + 1e-9:
                best_pos = best_candidate_pos
                best_r = best_candidate_r
                current_r = best_candidate_r
            
            consecutive_failures += 1
            
            if time.time() < deadline - 12.0 and consecutive_failures % 2 == 0:
                try:
                    for beta in [20.0, 40.0, 80.0]:
                        soft_pos, _ = _minimize_soft_min(best_pos, beta=beta, max_iter=150)
                        soft_r = _compute_radius(soft_pos)
                        if soft_r > best_r:
                            best_pos = soft_pos
                            best_r = soft_r
                            current_r = best_r
                            break
                    else:
                        polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=400)
                        polished_r = _compute_radius(polished)
                        if polished_r > best_r:
                            best_pos = polished
                            best_r = polished_r
                            current_r = best_r
                except Exception:
                    pass

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        for push_amount in [8e-6, 5e-6, 2e-6]:
            final_target = best_r + push_amount
            polished_pos, final_energy, _ = _minimize_energy(best_pos, final_target, max_iter=1000, ftol=1e-16, gtol=1e-15)
            final_r = _compute_radius(polished_pos)
            
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r
                break
        
        if time.time() < deadline - 1.0:
            polished_pos, _, _ = _minimize_energy(best_pos, best_r, max_iter=800, ftol=1e-16, gtol=1e-15)
            final_r = _compute_radius(polished_pos)
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r
        
        if time.time() < deadline - 0.5:
            try:
                soft_pos, _ = _minimize_soft_min(best_pos, beta=200.0, max_iter=300)
                soft_r = _compute_radius(soft_pos)
                if soft_r > best_r:
                    best_pos = soft_pos
                    best_r = soft_r
            except Exception:
                pass

    return best_pos
