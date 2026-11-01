import numpy as np
import time
from scipy.optimize import minimize

N = 100

# Precompute upper-triangle indices for pairwise distances
_iu_i, _iu_j = np.triu_indices(N, k=1)

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
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective_and_grad(x, beta, iu):
    """
    Soft-min objective for maximizing the minimum clearance.
    G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> -min_k c_k(x).
    Minimizing G(x) maximizes min_k c_k(x).
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
    # To avoid overflow/underflow, subtract max
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S  # (num_constraints,)
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    # Efficiently compute pair gradients
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]  # (N, N, 3)
    
    # Create weight matrix (N, N)
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights  # symmetric
    
    # grad[i] = - sum_j w_mat[i,j] * unit_diff[i,j]
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=10.0, beta_end=50000.0, n_steps=10):
    """
    Run L-BFGS-B optimization with increasing beta (soft-min to hard-min transition).
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = (_iu_i, _iu_j)
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
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x.copy()
        except Exception:
            pass
    
    return best_x, best_r


def _force_directed_refinement(centers, n_iters=200, deadline=None):
    """
    Force-directed method: spheres repel each other and are pushed away from walls.
    This is a physics-based local search that works well for packing problems.
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
        
        # Repulsion from other spheres (inverse square law, only for close pairs)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs within 3x current radius
        cutoff = 3.0 * max(r_current, 0.01)
        mask = dists < cutoff
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-8)
        
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, :, np.newaxis]
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Wall repulsion: push spheres away from walls
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < max(r_current, 0.01)
        wall_force = np.zeros_like(c)
        signs = np.sign(c)
        wall_force[wall_mask] = signs[wall_mask] * (1.0 / (wall_dists[wall_mask] + 1e-8))
        forces += wall_force
        
        # Weak centering force to keep spheres in the cube
        forces -= 0.0005 * c
        
        # Move with adaptive step size
        step = 0.001
        c_new = c + step * forces
        c_new = np.clip(c_new, -0.499, 0.499)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
            r_current = r_new
        else:
            # Reduce step size
            step *= 0.7
            if step < 1e-6:
                break
    
    return best_c, best_r


def _gradient_polish(centers, deadline, rng, n_iters=500):
    """
    Gradient-based local search using subgradient of the min function.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    step_size = 0.005
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Compute subgradient
        wall_clear = 0.5 - np.abs(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        pair_clear = dists[_iu_i, _iu_j] / 2.0
        
        all_clear = np.concatenate([wall_clear.ravel(), pair_clear])
        min_clear = np.min(all_clear)
        
        # Subgradient: average over all constraints within epsilon of the minimum
        eps = 1e-5
        mask = all_clear <= min_clear + eps
        n_active = np.sum(mask)
        if n_active == 0:
            break
        
        grad = np.zeros_like(c)
        
        # Wall contributions
        wall_mask = mask[:3*N]
        if np.any(wall_mask):
            signs = np.sign(c)
            wall_contrib = wall_mask.reshape(N, 3).astype(float) * (-signs)
            grad += wall_contrib / n_active
        
        # Pair contributions
        pair_mask = mask[3*N:]
        if np.any(pair_mask):
            active_pairs = np.where(pair_mask)[0]
            ai = _iu_i[active_pairs]
            aj = _iu_j[active_pairs]
            pair_diff = c[ai] - c[aj]
            pair_dists = dists[ai, aj]
            safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
            unit_vecs = pair_diff / safe_dists[:, np.newaxis]
            
            for k in range(len(ai)):
                grad[ai[k]] += unit_vecs[k] / n_active
                grad[aj[k]] -= unit_vecs[k] / n_active
        
        c_new = c + step_size * grad
        c_new = np.clip(c_new, -0.5, 0.5)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
            step_size = min(step_size * 1.1, 0.01)
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                # Add small random perturbation to escape local minimum
                idx = rng.integers(0, N)
                c = c.copy()
                c[idx] += rng.normal(0, 0.005, 3)
                c = np.clip(c, -0.5, 0.5)
                step_size = 0.003
    
    return best_c, best_r


def _annealing(centers, deadline, rng):
    """
    Simulated annealing with adaptive step sizes.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    T = 0.02
    T_min = 0.0005
    step_scale = 0.02
    cooling = 0.9995
    
    for it in range(15000):
        if time.time() > deadline:
            break
        
        # Perturb 1-3 random points
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        c_new = c.copy()
        c_new[idxs] += rng.normal(0, step_scale, size=(n_perturb, 3))
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_radius(c_new)
        delta = r_new - best_r
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        
        T = max(T * cooling, T_min)
    
    return best_c, best_r


def _init_grid(rng, nx, ny, nz, margin=0.02):
    """Grid initialization with small random perturbation."""
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_fcc(rng):
    """FCC-inspired initialization with two interleaved sublattices."""
    nx, ny, nz = 5, 5, 4
    centers = np.zeros((N, 3))
    idx = 0
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                
                sublattice = (ix + iy + iz) % 2
                if sublattice == 0:
                    offset = np.array([0.015, 0.015, 0.015])
                else:
                    offset = np.array([-0.015, -0.015, -0.015])
                
                centers[idx] = [x + offset[0], y + offset[1], z + offset[2]]
                idx += 1
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_random(rng):
    """Random initialization with minimum separation enforced."""
    centers = np.zeros((N, 3))
    for i in range(N):
        for attempt in range(2000):
            p = rng.uniform(-0.48, 0.48, size=3)
            if i == 0:
                centers[i] = p
                break
            dists = np.linalg.norm(centers[:i] - p, axis=1)
            if np.min(dists) > 0.08:
                centers[i] = p
                break
        else:
            centers[i] = rng.uniform(-0.48, 0.48, size=3)
    
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, 5s buffer
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(rng, 5, 5, 4, 0.02),
        _init_grid(rng, 4, 5, 5, 0.02),
        _init_grid(rng, 5, 4, 5, 0.02),
        _init_fcc(rng),
        _init_random(rng),
    ]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        c = init_centers.copy()
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Soft-min optimization (main structural improvement)
        if time.time() > deadline - 12.0:
            break
        c_opt, r_opt = _optimize_soft_min(c.ravel(), deadline - 10.0)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.reshape(-1, 3).copy()
        
        # Force-directed refinement
        if time.time() > deadline - 8.0:
            break
        c_fd, r_fd = _force_directed_refinement(best_centers, n_iters=100, deadline=deadline - 6.0)
        if r_fd > best_radius:
            best_radius = r_fd
            best_centers = c_fd.copy()
        
        # Gradient polish
        if time.time() > deadline - 5.0:
            break
        c_polished, r_polished = _gradient_polish(best_centers, deadline - 3.0, rng, n_iters=300)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = c_polished.copy()
    
    # Final refinement on the best solution
    if best_centers is not None and time.time() < deadline - 10.0:
        c = best_centers.copy()
        # More soft-min optimization with higher beta
        c2, r2 = _optimize_soft_min(c.ravel(), deadline - 5.0, beta_start=50.0, beta_end=100000.0, n_steps=6)
        if r2 > best_radius:
            best_radius = r2
            best_centers = c2.reshape(-1, 3).copy()
        
        # Final gradient polish
        if time.time() < deadline - 4.0:
            c3, r3 = _gradient_polish(best_centers, deadline - 2.0, rng, n_iters=500)
            if r3 > best_radius:
                best_radius = r3
                best_centers = c3.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(rng, 5, 5, 4, 0.02)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
