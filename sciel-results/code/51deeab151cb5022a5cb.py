"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective_and_grad(x, beta, iu):
    """
    Objective: soft-min approximation of the radius constraints.
    We minimize G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    which approximates -min(c_k) as beta->inf. Minimizing this maximizes min(c_k).
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = 0.5 - np.abs(c)  # (N, 3)
    wall_vals_flat = wall_vals.flatten()  # (3N,)
    
    # Pair constraints: c_p = dist_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_flat = pair_vals[iu]  # (num_pairs,)
    
    all_vals = np.concatenate([wall_vals_flat, pair_vals_flat])
    
    # Compute weights w_k = exp(-beta * c_k) / S
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S  # (num_constraints,)
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient
    pair_weights = weights[3*N:]
    
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]  # (N, N, 3)
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights
    
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=20.0, beta_end=20000.0, n_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_x,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _force_directed_refinement(centers, n_iters=200, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    best_c = c.copy()
    best_r = r_current
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Repulsion from other spheres
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Strong repulsion when close
        mask = dists < 2.5 * r_current
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-8)
        
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, np.newaxis]
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        wall_force = np.zeros_like(c)
        
        for axis in range(3):
            # Push away from walls
            wall_force[:, axis] += wall_mask[:, axis] * (1.0 / (np.abs(wall_dists[:, axis][wall_mask[:, axis]]) + 1e-8)) * np.sign(c[:, axis])
        
        forces -= wall_force
        
        # Update positions with adaptive step
        step = 0.01 * (1.0 + 0.5 * np.cos(it * 0.01))
        new_c = c + step * forces
        new_c = np.clip(new_c, -0.499999, 0.499999)
        
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            best_r = new_r
            best_c = new_c.copy()
            c = new_c
        else:
            c = new_c  # Still move, but keep best
    
    return best_c, best_r


def _parallel_tempering_mcmc(centers, deadline, n_temps=5, n_sweeps=500):
    """
    Parallel tempering MCMC to escape local minima.
    Uses different temperature replicas to explore the energy landscape.
    """
    best_c = centers.copy()
    best_r = _compute_radius(centers)
    
    # Define temperatures (log-spaced)
    temps = np.logspace(-2, 1, n_temps)  # 0.01 to 10
    
    # Initialize replicas
    replicas = []
    for t in temps:
        if t < 0.1:
            replicas.append(centers.copy())
        else:
            replicas.append(centers + np.random.normal(0, t * 0.01, (N, 3)))
    
    best_r_current = [0.0] * n_temps
    for i in range(n_temps):
        best_r_current[i] = _compute_radius(replicas[i])
    
    for sweep in range(n_sweeps):
        if time.time() > deadline:
            break
        
        # Update each replica
        for i in range(n_temps):
            c = replicas[i].copy()
            r_current = best_r_current[i]
            
            # Compute energy (negative radius)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Energy: sum of exp(-dist/tau) for close pairs + wall violations
            tau = 0.05
            pair_energy = np.sum(np.exp(-dists / tau))
            
            wall_dists = 0.5 - np.abs(c)
            wall_energy = np.sum(np.exp(-wall_dists / tau))
            
            energy = pair_energy + wall_energy
            
            # Propose move: random displacement
            idx = np.random.randint(N)
            delta = np.random.normal(0, 0.01 * temps[i], 3)
            c_new = c.copy()
            c_new[idx] += delta
            c_new[idx] = np.clip(c_new[idx], -0.499999, 0.499999)
            
            # Compute new energy
            diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
            dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
            np.fill_diagonal(dists_new, np.inf)
            pair_energy_new = np.sum(np.exp(-dists_new / tau))
            
            wall_dists_new = 0.5 - np.abs(c_new)
            wall_energy_new = np.sum(np.exp(-wall_dists_new / tau))
            energy_new = pair_energy_new + wall_energy_new
            
            # Metropolis acceptance
            if energy_new < energy or np.random.random() < np.exp(-(energy_new - energy) / temps[i]):
                replicas[i] = c_new
                r_new = _compute_radius(c_new)
                if r_new > best_r_current[i]:
                    best_r_current[i] = r_new
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c_new.copy()
        
        # Swap replicas
        for i in range(n_temps - 1):
            if np.random.random() < np.exp((best_r_current[i] - best_r_current[i+1]) * 10):
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                best_r_current[i], best_r_current[i+1] = best_r_current[i+1], best_r_current[i]
    
    return best_c, best_r


def _init_grid():
    """
    Initialize with a structured grid that approximates a dense packing.
    """
    # Use a 4x5x5 grid
    x_ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    y_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    
    grid = np.array([(x, y, z) for z in z_ticks for y in y_ticks for x in x_ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # 1. Initial Grid
    c0 = _init_grid()
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    # 2. Soft-min L-BFGS-B optimization
    if time.time() < timeout - 30:
        best_c, best_r = _optimize_soft_min(best_c, timeout - 30)
    
    # 3. Force-directed refinement
    if time.time() < timeout - 20:
        best_c, best_r = _force_directed_refinement(best_c, n_iters=150, deadline=timeout - 20)
    
    # 4. Parallel tempering MCMC for global exploration
    if time.time() < timeout - 10:
        best_c_mcmc, best_r_mcmc = _parallel_tempering_mcmc(best_c, deadline=timeout - 10, n_temps=4, n_sweeps=300)
        if best_r_mcmc > best_r:
            best_r = best_r_mcmc
            best_c = best_c_mcmc
    
    # 5. Final local polish with Nelder-Mead on exact objective
    if time.time() < timeout - 5:
        x0 = best_c.flatten()
        
        def exact_obj(x):
            c = x.reshape((N, 3))
            if not np.all(np.isfinite(c)):
                return 1e10
            if np.any(c > 0.5) or np.any(c < -0.5):
                return 1e10 + np.sum(np.abs(c - np.clip(c, -0.5, 0.5)))
            r = _compute_radius(c)
            return -r
        
        try:
            res = minimize(exact_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < exact_obj(x0):
                c_nm = res.x.reshape((N, 3))
                if np.all(np.isfinite(c_nm)) and np.all(c_nm <= 0.5) and np.all(c_nm >= -0.5):
                    r_nm = _compute_radius(c_nm)
                    if r_nm > best_r:
                        best_r = r_nm
                        best_c = c_nm
        except Exception:
            pass
    
    return best_c

# EVOLVE-BLOCK-END
