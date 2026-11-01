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
    r = min(wall_dists, pair_dists/2)
    We minimize S_beta = (1/beta) * ln(sum(exp(-beta * c_k)))
    which approximates -min(c_k) ... wait, we want to maximize min(c_k).
    So we minimize S_beta where S_beta approximates min(c_k) from above?
    No. S_beta(x) = (1/beta) ln(sum exp(-beta x_k)) -> min(x_k) as beta->inf.
    We want to maximize min(c_k). So we minimize -min(c_k).
    But -min(c_k) is not smooth.
    Let's minimize S_beta(-c_k) = (1/beta) ln(sum exp(beta c_k)) -> max(c_k) ... no.
    
    Let's stick to: minimize F(x) = (1/beta) * ln( sum_{k} exp(-beta * c_k(x)) )
    As beta -> inf, F(x) -> min_k c_k(x).
    So minimizing F(x) minimizes the minimum constraint value. That's wrong.
    We want to MAXIMIZE the minimum constraint value.
    So we should MINIMIZE -F(x)? No, F(x) is positive if c_k are positive.
    If c_k are positive, exp(-beta c_k) is small.
    Let's use: maximize min(c_k) <=> minimize max(-c_k).
    Soft-max of -c_k: (1/beta) ln(sum exp(beta * (-c_k))) = (1/beta) ln(sum exp(-beta c_k)).
    Yes! So we minimize G(x) = (1/beta) ln(sum_k exp(-beta c_k(x))).
    As beta -> inf, G(x) -> max_k (-c_k(x)) = -min_k c_k(x).
    So minimizing G(x) is equivalent to maximizing min_k c_k(x).
    
    Gradient:
    G(x) = (1/beta) ln(S), where S = sum exp(-beta c_k)
    dG/dx = (1/beta) * (1/S) * sum (-beta * exp(-beta c_k) * dc_k/dx)
    dG/dx = - (1/S) * sum exp(-beta c_k) * dc_k/dx
    dG/dx = - sum w_k * dc_k/dx, where w_k = exp(-beta c_k) / S
    
    This is a weighted average of the gradients of the constraints, with weights
    proportional to exp(-beta c_k). Constraints with smaller c_k (more violated)
    get higher weight.
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
    # Note: obj = (1/beta) * ln(sum exp(-beta c_k))
    # = (1/beta) * ln(sum exp(neg_beta_c))
    # = (1/beta) * (max_neg_beta_c + ln(sum exp(neg_beta_c - max_neg_beta_c)))
    # = (1/beta) * (max_neg_beta_c + ln(S))
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: d(c_w)/d(c_i) = -sign(c_i)
    # dG/d(c_i) = - sum_w w_w * d(c_w)/d(c_i) = - sum_w w_w * (-sign(c_i)) = sum_w w_w * sign(c_i)
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += np.sum(wall_weights * np.sign(c), axis=0) * 0 + np.sum(wall_weights, axis=0) * np.sign(c)
    # Wait, the formula is dG/dx = - sum w_k * dc_k/dx
    # For wall: dc_w/d(c_i) = -sign(c_i)
    # So dG/d(c_i) = - sum_w w_w * (-sign(c_i)) = sum_w w_w * sign(c_i)
    # But wall_weights is (N, 3), each weight corresponds to a specific (i, k)
    # So for c[i,k], the contribution is -w_{i,k} * (-sign(c[i,k])) = w_{i,k} * sign(c[i,k])
    # Let's be precise.
    # c_w[i,k] = 0.5 - |c[i,k]|
    # dc_w[i,k]/dc[i,k] = -sign(c[i,k])
    # dG/dc[i,k] = - w_{i,k} * (-sign(c[i,k])) = w_{i,k} * sign(c[i,k])
    
    # My previous line was wrong. Let's redo.
    grad = np.zeros_like(c)
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: d(c_p)/d(c_i) = (c_i - c_j) / dist_ij
    # dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    # We need to accumulate gradients for each i
    # For each pair (i,j) with weight w_ij:
    # grad[i] -= w_ij * (c_i - c_j) / dist_ij
    # grad[j] += w_ij * (c_i - c_j) / dist_ij  (since (c_j - c_i)/dist = -(c_i - c_j)/dist)
    
    # Efficiently:
    # unit_diff[i,j] = (c_i - c_j) / dist_ij
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]  # (N, N, 3)
    
    # Create weight matrix (N, N)
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights  # symmetric
    
    # grad[i] = - sum_j w_mat[i,j] * unit_diff[i,j]
    # = - sum_j w_mat[i,j] * (c_i - c_j) / dist_ij
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
        # Avoid division by zero
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
        _init_halton(rng),
        _init_random(rng),
    ]
    
    # Phase 1: Force-directed refinement + soft-min optimization
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
        
        # Soft-min optimization
        if time.time() > deadline - 10.0:
            break
        
        opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 5.0)
        opt_centers = opt_centers.reshape(-1, 3)
        
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
        
        # One more soft-min optimization with higher beta
        if time.time() < deadline - 5.0:
            opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 2.0, beta_start=500.0, beta_end=50000.0, n_steps=5)
            opt_centers = opt_centers.reshape(-1, 3)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
