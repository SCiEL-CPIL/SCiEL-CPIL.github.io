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
    Uses a smooth soft-penalty: quadratic when violated, zero otherwise.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms - smooth penalty for d_ij < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Only process pairs that are close enough to potentially violate
    close_mask = d_ij < 2.5 * R  # slightly wider margin for smooth transition
    if np.any(close_mask):
        i_close = i_idx[close_mask]
        j_close = j_idx[close_mask]
        d_close = d_ij[close_mask]
        
        # Smooth penalty: (max(0, 2R - d))^2
        violation = np.maximum(0.0, 2.0 * R - d_close)
        energy += np.sum(violation**2)
        
        # Gradient: d/dx_i sum (max(0, 2R - d_ij))^2
        # For pairs where 2R - d_ij > 0:
        # dE/dx_i = -2 * (2R - d_ij) * (x_i - x_j) / d_ij
        active = violation > 0
        if np.any(active):
            i_a = i_close[active]
            j_a = j_close[active]
            d_a = d_close[active]
            v_a = violation[active]
            
            factors = 2.0 * v_a / d_a
            vecs = centers[i_a] - centers[j_a]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_a, -forces)
            np.add.at(grad, j_a, forces)

    # 2. Wall terms - smooth penalty
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    wall_violation = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_violation**2)
    
    active_wall = wall_violation > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_violation[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val
            
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

def construct():
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

    # 2. SGLD Phase: Escape local minima using stochastic gradient Langevin dynamics
    # We run SGLD for a fraction of the time to explore the landscape,
    # then switch to deterministic optimization for refinement.
    
    # SGLD parameters
    T_start = 0.05  # Initial temperature
    T_end = 0.001   # Final temperature
    num_sgl_steps = 2000
    lr_sgl = 0.01   # Learning rate for SGLD
    
    # Reserve time for SGLD (e.g., 40% of remaining time)
    sgl_deadline = start_time + (deadline - start_time) * 0.4
    
    pos_sgl = best_pos.copy()
    
    # Anneal temperature over SGLD steps
    for step in range(num_sgl_steps):
        if time.time() > sgl_deadline:
            break
            
        # Linear annealing
        alpha = step / num_sgl_steps
        T_current = T_start * (1 - alpha) + T_end * alpha
        
        # Compute energy and gradient at current target radius (use current best_r as target for exploration)
        # Actually, for SGLD we want to minimize a potential that encourages large min-dist.
        # Let's use a fixed target slightly above current best to push for improvement, 
        # or simply minimize the energy for the current best_r to find a better local minimum.
        # Using current best_r is safer to keep particles valid.
        val, grad_flat = _energy_and_gradient(pos_sgl, best_r)
        grad = grad_flat.reshape(N, 3)
        
        # SGLD update: x_{t+1} = x_t - lr * grad + sqrt(2 * lr * T) * noise
        noise = rng.normal(0, 1, size=(N, 3))
        step_size_sgl = np.sqrt(2 * lr_sgl * T_current)
        
        pos_sgl = pos_sgl - lr_sgl * grad + step_size_sgl * noise
        
        # Project back to feasible region (box)
        pos_sgl = np.clip(pos_sgl, -0.5, 0.5)
        
        # Check if we found a better configuration
        # To save time, only check radius every 10 steps
        if step % 10 == 0:
            r_sgl = _compute_radius(pos_sgl)
            if r_sgl > best_r:
                best_r = r_sgl
                best_pos = pos_sgl.copy()
                # Restart SGLD from this new best to explore around it
                # Or just continue. Continuing is usually better for exploration.

    # 3. Deterministic Refinement Phase
    # Now use the remaining time for the original iterative radius increase + polish
    
    current_r = best_r
    step_size = 0.001
    
    consecutive_failures = 0
    max_failures = 4
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.6
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.003, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.01, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.03, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 400 if current_r < 0.11 else 600
            
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
            
            # If we failed, try to polish the current best position with a slightly smaller target
            # or just re-optimize the current best to ensure it's local optimal
            if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=300)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        # Try to push slightly beyond current best with very tight tolerances
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            target_r = best_r + 1e-6
            polished, energy, _ = _minimize_energy(best_pos, target_r, max_iter=2000, ftol=1e-16, gtol=1e-15)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_pos = polished
                best_r = polished_r
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
