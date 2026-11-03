"""Combined program: iterative radius increase with energy minimization."""
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
    return float(r)

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
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        sq_ij = sq_dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            # d/dx_i sum (2R - d_ij)^2 = -2(2R-d_ij) * d/dx_i d_ij
            # d/dx_i d_ij = (x_i - x_j)/d_ij
            # So grad_i = -2 v_ij * (x_i - x_j)/d_ij
            # But we are minimizing energy, so gradient is:
            # d(energy)/dx_i = sum_j -2(2R-d_ij) * (x_i-x_j)/d_ij
            # = -2 * sum_j v_ij * (x_i-x_j)/d_ij
            
            # Wait, let's re-derive.
            # E = sum (2R - d_ij)^2
            # dE/dx_i = sum_j 2(2R-d_ij) * (-1) * d(d_ij)/dx_i
            # = -2 sum_j (2R-d_ij) * (x_i - x_j)/d_ij
            
            # This pushes x_i away from x_j when d_ij < 2R.
            # (x_i - x_j)/d_ij is unit vector from j to i.
            # So force on i is in direction away from j. Correct.
            
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
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_val = max_abs[idx]
            max_k = np.argmax(abs_centers[idx])
            
            # dE/dx_ik = 2(R - w_k) * (-1) * d(w_k)/dx_ik
            # w_k = 0.5 - |x_ik|
            # d(w_k)/dx_ik = -sign(x_ik)
            # dE/dx_ik = -2 v * (-sign(x_ik)) = 2 v sign(x_ik)
            
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

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

def _soft_min_objective_and_grad(x, temperature=0.001):
    """Soft-min radius objective for final polishing."""
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        exp_terms = np.zeros_like(sq_dists)
        exp_terms[finite_mask] = np.exp(-(sq_dists[finite_mask] - max_sq) / (4.0 * T2))
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq_pair = max_sq / 4.0 - T2 * log_sum
        else:
            soft_min_sq_pair = max_sq / 4.0
    else:
        soft_min_sq_pair = 0.0
    
    soft_min_sq_pair = max(soft_min_sq_pair, 0.0)
    soft_min_pair = np.sqrt(soft_min_sq_pair)
    
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    grad = np.zeros_like(centers)
    
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_soft_min_wall_dx = -exp_wall * signs / sum_wall
        
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        grad = grad_combined
    
    return -soft_min_combined, -grad.flatten()

def _soft_min_optimize(init_centers, temperature=0.001, max_iter=3000, time_limit=10.0):
    """Optimize using soft-min objective."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        return _soft_min_objective_and_grad(x, temperature)
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'maxfun': 8000}
    )
    
    return result.x.reshape(N, 3)

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            push = (r - dist/2.0) * d / dist * 0.5
            centers[i] += push
            centers[j] -= push
        else:
            i, k = min_wall_idx
            wall = wall_dists[i, k]
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # 1. Initialization: 5x4x5 Grid with symmetry breaking
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    # Add small noise to break symmetry
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial optimization to get a good baseline
    # Start with a conservative radius
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.0005  # Smaller steps for stability
    
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
                    # Don't mark as full success for step size increase, but accept the position
                    pass
        
        if not success:
            consecutive_failures += 1
            
            # If we failed, try to polish the current best
            if consecutive_failures == max_failures:
                # Reset step size and try polishing
                step_size = 0.0005
                polished = _polish_critical_pairs(best_pos, time_limit=min(2.0, deadline - time.time() - 5.0))
                r_polished = _compute_radius(polished)
                if r_polished > best_r:
                    best_pos = polished
                    best_r = r_polished
                    current_r = r_polished

    # 3. Fine polishing with soft-min from best found
    if time.time() < deadline - 10.0:
        fine_temps = [0.001, 0.0005, 0.0002, 0.0001]
        for temp in fine_temps:
            if time.time() > deadline - 5.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 3.0
                if time_remaining < 2.0:
                    break
                optimized = _soft_min_optimize(best_pos, temperature=temp,
                                                max_iter=2000, time_limit=min(3.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pos = optimized.copy()
            except Exception:
                continue
    
    # 4. Final high-precision polish using L-BFGS with very small temperature
    if time.time() < deadline - 3.0:
        try:
            time_remaining = deadline - time.time() - 1.0
            if time_remaining > 1.0:
                optimized = _soft_min_optimize(best_pos, temperature=0.00001,
                                                max_iter=3000, time_limit=min(2.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pos = optimized.copy()
        except Exception:
            pass
    
    # 5. Exact local polish on critical pairs
    if time.time() < deadline - 1.0:
        try:
            time_remaining = deadline - time.time() - 0.5
            if time_remaining > 0.5:
                polished = _polish_critical_pairs(best_pos, time_limit=min(0.5, time_remaining))
                r_polished = _compute_radius(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_pos = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_pos is None:
        best_pos = grid[:N].copy()
    
    best_pos = np.clip(best_pos, -0.4999999, 0.4999999)
    
    return best_pos
