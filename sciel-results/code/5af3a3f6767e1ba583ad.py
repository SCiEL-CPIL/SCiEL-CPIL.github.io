import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _penalty_objective_and_grad(x, r_target, mu):
    """
    Penalty method objective: minimize sum of constraint violations + mu * (r - r_target)^2
    where constraints are:
      - Wall: 0.5 - |c_ik| >= r_target
      - Pair: |c_i - c_j|/2 >= r_target
    
    x = [c_1, ..., c_N] (3N values)
    We want to maximize r, so we fix r_target and minimize violations.
    
    Objective: sum_k max(0, violation_k)^2 + mu * (r_target - r_actual)^2
    
    Actually, let's reformulate: minimize F(x) = sum of squared constraint violations
    where violation_k = max(0, r_target - constraint_k)
    
    constraint_k for walls: 0.5 - |c_ik|
    constraint_k for pairs: |c_i - c_j| / 2
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c[i,k]| >= r_target
    wall_vals = 0.5 - np.abs(c)  # (N, 3)
    wall_violations = np.maximum(0, r_target - wall_vals)  # (N, 3)
    
    # Pair constraints: |c_i - c_j|/2 >= r_target
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0  # (N, N)
    iu_i, iu_j = np.triu_indices(N, k=1)
    pair_violations = np.maximum(0, r_target - pair_vals[iu_i, iu_j])  # (num_pairs,)
    
    # Objective: sum of squared violations
    obj = np.sum(wall_violations ** 2) + np.sum(pair_violations ** 2)
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: d/d(c[i,k]) max(0, r_target - (0.5 - |c[i,k]|))^2
    # = 2 * max(0, r_target - (0.5 - |c[i,k]|)) * (-d/d(c[i,k]) (0.5 - |c[i,k]|))
    # = -2 * max(0, r_target - (0.5 - |c[i,k]|)) * (-sign(c[i,k]))
    # = 2 * max(0, r_target - (0.5 - |c[i,k]|)) * sign(c[i,k])
    wall_grad = 2.0 * wall_violations * np.sign(c)
    grad += wall_grad
    
    # Pair gradient: d/d(c[i]) max(0, r_target - |c_i-c_j|/2)^2
    # For pair (i,j) with violation v_ij = max(0, r_target - dist_ij/2):
    # d/d(c[i]) v_ij^2 = 2 * v_ij * (-1/2) * d(dist_ij)/d(c[i])
    # = -v_ij * (c[i] - c[j]) / dist_ij
    # d/d(c[j]) v_ij^2 = -v_ij * (c[j] - c[i]) / dist_ij
    # = v_ij * (c[i] - c[j]) / dist_ij
    
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    
    # Only upper triangle matters
    pair_viol_mat = np.zeros((N, N))
    pair_viol_mat[iu_i, iu_j] = pair_violations
    pair_viol_mat[iu_j, iu_i] = pair_violations  # symmetric
    
    # grad[i] = sum_j (-pair_viol_mat[i,j] * (c[i]-c[j]) / dists[i,j])
    unit_diff = diff * inv_dists[:, :, np.newaxis]  # (N,N,3)
    grad_pairs = -np.sum(pair_viol_mat[:, :, np.newaxis] * unit_diff, axis=1)
    
    grad += grad_pairs
    
    return obj, grad.ravel()


def _optimize_with_penalty(x0, r_target, mu_schedule, deadline):
    """
    Run penalty method optimization.
    x0: initial center positions (3N,)
    r_target: target radius to achieve
    mu_schedule: schedule of penalty multipliers
    deadline: time limit
    """
    best_x = x0.copy()
    best_obj = np.inf
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for mu in mu_schedule:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _penalty_objective_and_grad,
                best_x,
                args=(r_target, mu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_obj, _ = _penalty_objective_and_grad(new_x, r_target, mu)
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = new_x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    return best_x


def _binary_search_radius(centers_init, deadline):
    """
    Binary search for the maximum achievable radius.
    For each candidate r, try to find positions where all constraints are satisfied.
    """
    c0 = centers_init.copy()
    r_low = _compute_radius(c0)
    r_high = 0.12  # Upper bound estimate
    
    best_centers = c0.copy()
    best_r = r_low
    
    mu_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    # Binary search
    for _ in range(15):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to achieve r_mid
        x_opt = _optimize_with_penalty(c0.flatten(), r_mid, mu_schedule, deadline - 3.0)
        x_opt = np.clip(x_opt, -0.5, 0.5)
        
        r_achieved = _compute_radius(x_opt.reshape(-1, 3))
        
        if r_achieved >= r_mid - 1e-6:
            # Can achieve r_mid, try higher
            r_low = r_mid
            best_r = r_achieved
            best_centers = x_opt.reshape(-1, 3)
        else:
            # Cannot achieve r_mid, try lower
            r_high = r_mid
    
    return best_centers, best_r


def _force_directed_refinement(centers, n_iters=100, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    More stable than greedy push-apart.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Repulsion from other spheres
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                # Strong repulsion when close
                if dist < 2.5 * r_current:
                    force_mag = 1.0 / (dist ** 2 + 1e-8)
                    unit = diff / dist
                    forces[i] += force_mag * unit
                    forces[j] -= force_mag * unit
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                wall_dist = 0.5 - abs(c[i, k])
                if wall_dist < r_current:
                    # Push inward
                    if c[i, k] > 0:
                        forces[i, k] -= 0.5 / (wall_dist + 1e-8)
                    else:
                        forces[i, k] += 0.5 / (wall_dist + 1e-8)
        
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
    # FCC in a cube: place at corners and face centers of sub-cubes
    # Use 4x4x4 = 64 positions, then add 36 more
    centers = []
    # FCC positions in unit cell
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]
    ]
    
    nx = ny = nz = 4
    sub_size = 1.0 / nx
    
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
        # Fill remaining with random
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


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
        _init_random(rng),
    ]
    
    # Phase 1: Force-directed refinement + penalty optimization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        c = init_centers.copy()
        
        # Force-directed refinement
        c = _force_directed_refinement(c, n_iters=50, deadline=deadline - 10.0)
        
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Penalty optimization with binary search on radius
        if time.time() > deadline - 10.0:
            break
        
        opt_centers, opt_r = _binary_search_radius(c, deadline - 5.0)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers.copy()
    
    # Phase 2: Final refinement on best result
    if best_centers is not None and time.time() < deadline - 10.0:
        c = best_centers.copy()
        c = _force_directed_refinement(c, n_iters=80, deadline=deadline - 5.0)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
        
        # One more penalty optimization
        if time.time() < deadline - 5.0:
            opt_centers, opt_r = _binary_search_radius(c, deadline - 2.0)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
