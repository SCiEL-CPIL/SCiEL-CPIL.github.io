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
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_abs
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
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)

    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R

    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]

        energy += np.sum(v_ij**2)

        # Gradient of sum (2R - d)^2 w.r.t centers
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs

        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs

    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]

        energy += np.sum(violations_wall**2)

        # Vectorized wall gradient
        # The active constraint is the one with the max absolute value.
        # Gradient points away from the nearest wall.
        max_k = np.argmax(abs_centers[i_valid_wall], axis=1)
        signs = np.sign(centers[i_valid_wall, max_k])
        grad_vals = 2.0 * violations_wall * signs
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
    # d(soft_min)/d(pos_i) = sum_j w_j * d(d_ij/2)/d(pos_i)
    # d(d_ij)/d(pos_i) = (pos_i - pos_j) / d_ij
    # So d(soft_min)/d(pos_i) = sum_j w_j * (pos_i - pos_j) / (2 * d_ij)

    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]

    # Vectorized accumulation
    # i_idx, j_idx
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
    FCC is denser than simple cubic, providing a better starting point.
    """
    # FCC unit cell has 4 atoms. We need 100 spheres.
    # 25 unit cells * 4 = 100.
    # Let's create a grid of unit cells.
    # 5x5x1 grid of FCC unit cells? Or 4x4x2?
    # Let's use a 5x5x1 grid of FCC cells to fill the cube roughly.
    # Actually, a 4x4x2 grid of FCC cells gives 4*4*2*4 = 128 atoms. Too many.
    # A 5x4x1 grid gives 5*4*1*4 = 80 atoms. Too few.
    # A 4x5x1 grid gives 80.
    # A 5x5x1 grid gives 100 atoms exactly.
    # FCC lattice vectors:
    # a1 = (0.5, 0.5, 0)
    # a2 = (0.5, 0, 0.5)
    # a3 = (0, 0.5, 0.5)
    # Basis points: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    
    # We want to fit 100 points in [-0.5, 0.5]^3.
    # Side length 1.
    # 5x5x1 grid of FCC cells.
    # Cell size a. 5a = 1 => a = 0.2.
    # FCC cell has 4 atoms.
    # Total atoms = 5 * 5 * 1 * 4 = 100.
    
    a = 0.2
    positions = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(1):
                # Origin of cell
                ox = -0.5 + ix * a
                oy = -0.5 + iy * a
                oz = -0.5 + iz * a
                
                # Basis points
                # (0,0,0)
                positions.append([ox, oy, oz])
                # (0.5, 0.5, 0)
                positions.append([ox + a/2, oy + a/2, oz])
                # (0.5, 0, 0.5)
                positions.append([ox + a/2, oy, oz + a/2])
                # (0, 0.5, 0.5)
                positions.append([ox, oy + a/2, oz + a/2])
    
    pos = np.array(positions)
    # Add small noise to break symmetry
    rng = np.random.default_rng(42)
    pos += rng.normal(0, 0.01, size=pos.shape)
    pos = np.clip(pos, -0.49, 0.49)
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Initialization: FCC lattice
    best_pos = _generate_initial_positions()
    
    # Initial optimization to get a good baseline
    # Use soft-min to maximize the radius directly
    # Start with a moderate beta
    beta_start = 50.0
    best_pos, _ = _minimize_soft_min(best_pos, beta_start, max_iter=500)
    best_r = _compute_radius(best_pos)

    if best_r <= 0:
        rng = np.random.default_rng(42)
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    # We will alternate between increasing target radius with energy minimization
    # and polishing with soft-min.
    
    current_r = best_r
    step_size = 0.0005  # Smaller steps for stability

    consecutive_failures = 0
    max_failures = 5

    rng_search = np.random.default_rng(123)

    # Main loop: try to increase radius
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size

        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break

        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.005, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.02, size=(N, 3)))

        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)

            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 300 if current_r < 0.11 else 500

            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)

            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-6:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not perfectly zero, if we improved the radius, we can take it
                actual_r = _compute_radius(new_pos)
                if actual_r > current_r:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    # Don't mark as full success, but accept the position
                    consecutive_failures = 0
                    break
                else:
                    consecutive_failures += 1

        if not success:
            consecutive_failures += 1

    # 3. Final high-precision polish using soft-min surrogate
    if time.time() < deadline - 5.0:
        # Try multiple beta values for the soft-min polish
        # Higher beta makes the soft-min closer to the true min
        betas = [100.0, 200.0, 500.0]
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            t_left = deadline - time.time() - 3.0
            if t_left < 2.0:
                break

            polished, _ = _minimize_soft_min(best_pos, beta, max_iter=1000)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_pos = polished
                best_r = r_polished
                current_r = r_polished

        # Final energy-based polish with tight tolerances
        if time.time() < deadline - 3.0:
            final_pos, _, _ = _minimize_energy(best_pos, current_r, max_iter=1000, ftol=1e-15, gtol=1e-13)
            r_final = _compute_radius(final_pos)
            if r_final > best_r:
                best_pos = final_pos
                best_r = r_final

    # Ensure valid output
    best_pos = np.clip(best_pos, -0.5, 0.5)
    if not np.all(np.isfinite(best_pos)):
        # Fallback to grid
        best_pos = _generate_initial_positions()

    return best_pos
