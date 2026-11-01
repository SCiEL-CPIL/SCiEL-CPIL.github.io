import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

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

def _identify_critical_constraints(centers, r_current):
    """
    Identify which constraints are critical (tightest) for the current radius.
    Returns:
        wall_mask: boolean array of shape (N*3,) - True for critical wall coords
        pair_indices: list of (i, j) pairs that are critical
    """
    c = centers
    wall_dists = 0.5 - np.abs(c)
    # A wall constraint is critical if it's within 1% of the current radius
    wall_mask = wall_dists <= r_current * 1.01
    
    # Pair constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # A pair constraint is critical if it's within 1% of the current radius
    pair_mask = pair_dists <= r_current * 1.01
    pair_indices = np.argwhere(pair_mask & (np.arange(N)[:, None] < np.arange(N)[None, :]))
    
    return wall_mask, pair_indices


def _soft_min_objective_and_grad(x, beta, iu):
    """
    Objective: soft-min approximation of the radius constraints.
    G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> max_k (-c_k(x)) = -min_k c_k(x).
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

def _focused_soft_min_objective_and_grad(x, beta, iu, critical_wall_mask, critical_pair_indices):
    """
    Focused soft-min: only optimize over critical constraints.
    This reduces the number of constraints in the log-sum-exp, making the
    objective more focused on the binding constraints.
    """
    c = x.reshape(-1, 3)
    
    # Only include critical wall constraints
    wall_vals = 0.5 - np.abs(c)
    wall_vals_flat = wall_vals.flatten()
    critical_wall_vals = wall_vals_flat[critical_wall_mask]
    
    # Only include critical pair constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    if len(critical_pair_indices) > 0:
        critical_pair_vals = np.array([pair_vals[i, j] for i, j in critical_pair_indices])
    else:
        critical_pair_vals = np.array([1.0])  # dummy to avoid empty
    
    all_vals = np.concatenate([critical_wall_vals, critical_pair_vals])
    
    # Compute weights
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient for critical walls only
    wall_weights = weights[:len(critical_wall_vals)]
    # Reconstruct full wall weight array
    full_wall_weights = np.zeros(3*N)
    full_wall_weights[critical_wall_mask] = wall_weights
    grad += full_wall_weights.reshape(-1, 3) * np.sign(c)
    
    # Pair gradient for critical pairs only
    pair_weights = weights[len(critical_wall_vals):]
    
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    # Only add gradients for critical pairs
    if len(critical_pair_indices) > 0:
        for idx, (i, j) in enumerate(critical_pair_indices):
            w = pair_weights[idx]
            if w > 1e-15:
                grad[i] -= w * unit_diff[i, j]
                grad[j] += w * unit_diff[j, i]
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=20.0, beta_end=20000.0, n_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta.
    Uses focused optimization on critical constraints in later stages.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for step_idx, beta in enumerate(betas):
        if time.time() > deadline:
            break
        
        # Determine if we should use focused optimization
        # Use focused optimization when beta is high (later steps)
        use_focused = beta > 1000.0
        
        if use_focused:
            # Identify critical constraints
            c_current = best_x.reshape(-1, 3)
            critical_wall_mask, critical_pair_indices = _identify_critical_constraints(c_current, best_r)
            
            # If no critical constraints found, fall back to full optimization
            if np.sum(critical_wall_mask) == 0 and len(critical_pair_indices) == 0:
                use_focused = False
        
        try:
            if use_focused:
                res = minimize(
                    _focused_soft_min_objective_and_grad,
                    best_x,
                    args=(beta, iu, critical_wall_mask, critical_pair_indices),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 40}
                )
            else:
                res = minimize(
                    _soft_min_objective_and_grad,
                    best_x,
                    args=(beta, iu),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
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
    Uses cKDTree for efficient neighbor queries.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces using KDTree for efficiency
        forces = np.zeros_like(c)
        
        # Build tree and query neighbors within cutoff
        tree = cKDTree(c)
        cutoff = 2.5 * r_current
        pairs = tree.query_pairs(cutoff, output_type='ndarray')
        
        if len(pairs) > 0:
            for i, j in pairs:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    force_mag = 1.0 / (dist ** 2 + 1e-8)
                    unit_diff = diff / dist
                    forces[i] += force_mag * unit_diff
                    forces[j] -= force_mag * unit_diff
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        if np.any(wall_mask):
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
    Uses cKDTree for efficient neighbor queries in the inner loop.
    """
    c = centers.copy()
    r = _compute_radius(c)
    r_target = r * 1.001
    
    for outer in range(n_outer):
        if time.time() > deadline:
            break
        
        # Build KDTree once per outer iteration
        tree = cKDTree(c)
        
        def energy_and_grad(x):
            cx = x.reshape(-1, 3)
            cx = np.clip(cx, -0.5 + 1e-10, 0.5 - 1e-10)
            
            # Use tree for pair interactions - only consider close pairs
            # But we need gradient, so compute full pairwise for correctness
            # However, we can skip distant pairs in the energy computation
            diff = cx[:, np.newaxis, :] - cx[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Overlap penalty: max(0, 2r - d)^2
            overlap = np.maximum(0, 2 * r_target - dists)
            E_pair = np.sum(overlap ** 2)
            
            # Gradient of pair term
            grad_pair = np.zeros_like(cx)
            mask = overlap > 1e-10
            if np.any(mask):
                unit_diff = diff / dists[:, :, np.newaxis]
                unit_diff_masked = np.where(mask[:, :, np.newaxis], unit_diff, 0)
                overlap_sym = overlap.copy()
                grad_pair = 2 * np.sum(overlap_sym[:, :, np.newaxis] * unit_diff_masked, axis=1)
            
            # Wall penalty
            wall_dist = 0.5 - np.abs(cx)
            wall_viol = np.maximum(0, r_target - wall_dist)
            E_wall = np.sum(wall_viol ** 2)
            
            grad_wall = np.zeros_like(cx)
            wall_mask = wall_viol > 1e-10
            grad_wall[wall_mask] = -2 * wall_viol[wall_mask] * np.sign(cx[wall_mask])
            
            E = E_pair + E_wall
            grad = grad_pair + grad_wall
            
            return E, grad.ravel()
        
        # Run L-BFGS-B
        try:
            res = minimize(
                energy_and_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
        except Exception:
            pass
        
        # Update r
        new_r = _compute_radius(c)
        if new_r > r:
            r = new_r
            r_target = r * 1.001
    
    return c


def _kdtree_polish(centers, deadline, n_iters=50):
    """
    Final high-precision polish using cKDTree for efficient neighbor identification.
    Identifies the critical (tightest) constraints and focuses optimization there.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Build tree and find close pairs
        tree = cKDTree(c)
        cutoff = 2.2 * r_current
        pairs = tree.query_pairs(cutoff, output_type='ndarray')
        
        if len(pairs) == 0:
            break
        
        # Compute forces only from close pairs
        forces = np.zeros_like(c)
        for i, j in pairs:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                # Stronger repulsion for closer pairs
                force_mag = 1.0 / (dist ** 3 + 1e-10)
                unit_diff = diff / dist
                forces[i] += force_mag * unit_diff
                forces[j] -= force_mag * unit_diff
        
        # Wall forces
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current * 1.1
        if np.any(wall_mask):
            wall_force = np.zeros_like(c)
            wall_force[wall_mask] = 1.0 / (wall_dists[wall_mask] ** 2 + 1e-10)
            forces[wall_mask] -= wall_force[wall_mask] * np.sign(c[wall_mask])
        
        # Small adaptive step
        max_force = np.max(np.abs(forces))
        if max_force > 1e-12:
            step = 0.005 / max_force
        else:
            step = 0.005
        
        c += step * forces
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
    
    return c

def _high_precision_polish(centers, deadline, n_outer=10):
    """
    High-precision polish: alternate between focused soft-min optimization and
    force-directed refinement with very tight tolerances.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for outer in range(n_outer):
        if time.time() > deadline:
            break
        
        # Step 1: Focused soft-min optimization with very high beta
        beta = 50000.0
        critical_wall_mask, critical_pair_indices = _identify_critical_constraints(c, r_current)
        
        if np.sum(critical_wall_mask) > 0 or len(critical_pair_indices) > 0:
            try:
                res = minimize(
                    _focused_soft_min_objective_and_grad,
                    c.flatten(),
                    args=(beta, iu, critical_wall_mask, critical_pair_indices),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
                )
                c = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
            except Exception:
                pass
        
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
        
        # Step 2: Very fine force-directed polish
        c = _kdtree_polish(c, deadline - 2.0, n_iters=20)
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
    
    return c


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
    
    # Phase 1: Force-directed refinement + soft-min optimization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        c = init_centers.copy()
        
        # Force-directed refinement
        c = _force_directed_refinement(c, n_iters=40, deadline=deadline - 20.0)
        
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Soft-min optimization
        if time.time() > deadline - 18.0:
            break
        
        opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 12.0)
        opt_centers = opt_centers.reshape(-1, 3)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers.copy()
    
    # Phase 2: Potential energy minimization on best result
    if best_centers is not None and time.time() < deadline - 25.0:
        c = best_centers.copy()
        c = _potential_energy_minimization(c, deadline - 18.0, n_outer=30)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
    
    # Phase 3: KDTree-based polish on best result
    if best_centers is not None and time.time() < deadline - 15.0:
        c = best_centers.copy()
        c = _kdtree_polish(c, deadline - 8.0, n_iters=60)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
        
        # One more soft-min optimization with higher beta
        if time.time() < deadline - 5.0:
            opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 2.0, beta_start=500.0, beta_end=50000.0, n_steps=5)
            opt_centers = opt_centers.reshape(-1, 3)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
        
        # Final KDTree polish
        if time.time() < deadline - 1.0:
            c = best_centers.copy()
            c = _kdtree_polish(c, deadline - 0.5, n_iters=30)
            r = _compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_centers = c
    
    # Phase 4: High-precision polish with focused optimization
    if best_centers is not None and time.time() < deadline - 3.0:
        c = best_centers.copy()
        c = _high_precision_polish(c, deadline - 0.5, n_outer=8)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
