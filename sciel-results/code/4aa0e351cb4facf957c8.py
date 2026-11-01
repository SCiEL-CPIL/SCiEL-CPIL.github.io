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
    G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> -min_k c_k(x).
    Minimizing G(x) maximizes min_k c_k(x).
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = 0.5 - np.abs(c)
    wall_vals_flat = wall_vals.flatten()
    
    # Pair constraints: c_p = dist_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_flat = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals_flat, pair_vals_flat])
    
    # Compute weights w_k = exp(-beta * c_k) / S
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights
    
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=20.0, beta_end=20000.0, n_steps=8):
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
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _force_directed_refinement(centers, n_iters=100, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
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
        
        unit_diff = diff / dists[:, :, np.newaxis]
        unit_diff[~mask] = 0
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        wall_force = np.zeros_like(c)
        wall_force[wall_mask] = 0.5 / (wall_dists[wall_mask] + 1e-8)
        forces -= wall_force * np.sign(c)
        
        # Adaptive step size
        max_force = np.max(np.abs(forces))
        if max_force > 1e-12:
            step = 0.01 / max_force
        else:
            step = 0.01
        
        c += step * forces
        c = np.clip(c, -0.499, 0.499)
        
        # Check if improved
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
    
    return c


def _simulated_annealing(x0, deadline, beta=500.0, n_iterations=200, iu=None):
    """
    Simulated annealing using the soft-min objective.
    Escapes local minima by accepting worse solutions with probability exp(-delta/E).
    """
    if iu is None:
        iu = np.triu_indices(N, k=1)
    
    c = x0.reshape(-1, 3).copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    best_obj, _ = _soft_min_objective_and_grad(c.flatten(), beta, iu)
    
    # Temperature schedule
    T_start = 0.01
    T_end = 0.0001
    n_inner = max(1, n_iterations // 10)
    
    for outer in range(10):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (outer / 10.0)
        
        for it in range(n_inner):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            rng = np.random.default_rng(hash((outer, it)) % (2**32))
            perturbation_scale = 0.01 * (1.0 - outer / 10.0) + 0.001
            perturbation = rng.normal(0, perturbation_scale, c.shape)
            
            c_new = np.clip(c + perturbation, -0.5, 0.5)
            
            # Compute new objective
            try:
                obj_new, _ = _soft_min_objective_and_grad(c_new.flatten(), beta, iu)
            except Exception:
                continue
            
            delta_obj = obj_new - best_obj
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_obj < 0 or np.random.random() < np.exp(-delta_obj / T):
                c = c_new
                best_obj = obj_new
                
                r_new = _compute_radius(c)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c.copy()
    
    return best_c.flatten(), best_r


def _init_grid(nx, ny, nz, rng):
    """Create a perturbed grid initialization."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    noise_scale = 0.008
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_fcc_like(rng):
    """FCC-like initialization."""
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]
    ]
    
    nx = ny = nz = 4
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in fcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_bcc_like(rng):
    """BCC-like initialization."""
    bcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0.5]
    ]
    
    nx = ny = nz = 5
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in bcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_random(rng):
    """Random initialization."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _init_halton(rng):
    """Halton sequence initialization."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    centers = np.zeros((N, 3))
    for i in range(1, N + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _potential_energy_minimization(centers, deadline, n_outer=50):
    """
    Minimize a smooth potential energy that penalizes overlaps and wall violations.
    """
    c = centers.copy()
    r = _compute_radius(c)
    r_target = r * 1.001
    
    for outer in range(n_outer):
        if time.time() > deadline:
            break
        
        def energy_and_grad(x):
            cx = x.reshape(-1, 3)
            cx = np.clip(cx, -0.5 + 1e-10, 0.5 - 1e-10)
            
            diff = cx[:, np.newaxis, :] - cx[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            overlap = np.maximum(0, 2 * r_target - dists)
            E_pair = np.sum(overlap ** 2)
            
            grad_pair = np.zeros_like(cx)
            mask = overlap > 1e-10
            if np.any(mask):
                unit_diff_masked = np.where(mask[:, :, np.newaxis], diff / dists[:, :, np.newaxis], 0)
                overlap_sym = overlap.copy()
                grad_pair = 2 * np.sum(overlap_sym[:, :, np.newaxis] * unit_diff_masked, axis=1)
            
            wall_dist = 0.5 - np.abs(cx)
            wall_viol = np.maximum(0, r_target - wall_dist)
            E_wall = np.sum(wall_viol ** 2)
            
            grad_wall = np.zeros_like(cx)
            wall_mask = wall_viol > 1e-10
            grad_wall[wall_mask] = -2 * wall_viol[wall_mask] * np.sign(cx[wall_mask])
            
            E = E_pair + E_wall
            grad = grad_pair + grad_wall
            
            return E, grad.ravel()
        
        try:
            res = minimize(
                energy_and_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
        except Exception:
            pass
        
        new_r = _compute_radius(c)
        if new_r > r:
            r = new_r
            r_target = r * 1.001
    
    return c


def _high_precision_polish(x0, deadline):
    """
    Final high-precision polish with very tight tolerances and high beta.
    """
    iu = np.triu_indices(N, k=1)
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    
    # Very high beta for near-exact min
    betas = [10000, 50000, 100000, 200000]
    
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
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(5, 5, 4, rng),
        _init_grid(4, 5, 5, rng),
        _init_grid(5, 4, 5, rng),
        _init_fcc_like(rng),
        _init_bcc_like(rng),
        _init_halton(rng),
        _init_random(rng),
    ]
    
    # Phase 1: Force-directed refinement + soft-min optimization + SA
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        c = init_centers.copy()
        
        # Force-directed refinement
        c = _force_directed_refinement(c, n_iters=50, deadline=deadline - 25.0)
        
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Soft-min optimization
        if time.time() > deadline - 25.0:
            break
        
        opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 20.0)
        opt_centers = opt_centers.reshape(-1, 3)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers.copy()
        
        # Simulated annealing to escape local minima
        if time.time() > deadline - 20.0:
            break
        
        sa_centers, sa_r = _simulated_annealing(best_centers.flatten(), deadline - 15.0, beta=500.0, n_iterations=150)
        sa_centers = sa_centers.reshape(-1, 3)
        
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
    
    # Phase 2: Potential energy minimization on best result
    if best_centers is not None and time.time() < deadline - 25.0:
        c = best_centers.copy()
        c = _potential_energy_minimization(c, deadline - 20.0, n_outer=40)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
    
    # Phase 3: More SA + refinement cycles
    if best_centers is not None and time.time() < deadline - 15.0:
        for cycle in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # SA to explore
            sa_centers, sa_r = _simulated_annealing(best_centers.flatten(), deadline - 8.0, beta=1000.0, n_iterations=100)
            sa_centers = sa_centers.reshape(-1, 3)
            
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()
            
            # Refine with soft-min
            if time.time() > deadline - 8.0:
                break
            
            opt_centers, opt_r = _optimize_soft_min(best_centers.flatten(), deadline - 5.0, beta_start=500.0, beta_end=50000.0, n_steps=5)
            opt_centers = opt_centers.reshape(-1, 3)
            
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers.copy()
    
    # Phase 4: Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5.0:
        c = best_centers.copy()
        c = _force_directed_refinement(c, n_iters=50, deadline=deadline - 3.0)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
        
        # High-precision polish
        if time.time() < deadline - 2.0:
            polished, polished_r = _high_precision_polish(best_centers.flatten(), deadline - 0.5)
            polished = polished.reshape(-1, 3)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
