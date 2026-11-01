import numpy as np
import time
from scipy.optimize import minimize

N = 100

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
    As beta -> inf, G(x) -> max_k (-c_k(x)) = -min_k c_k(x).
    So minimizing G(x) maximizes min_k c_k(x).
    
    c_k are the constraint values: wall_dists and pair_dists/2.
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
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
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


def _force_directed_refinement(centers, n_iters=200, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    best_r = r_current
    best_c = c.copy()
    
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
        unit_diff[mask] = diff[mask] / dists[mask, :, np.newaxis]
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        for i in range(N):
            for k in range(3):
                if wall_mask[i, k]:
                    forces[i, k] += np.sign(c[i, k]) * 5.0 / (wall_dists[i, k] + 1e-8)
        
        # Update positions
        c += 0.01 * forces
        c = np.clip(c, -0.49, 0.49)
        
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r


def _init_grid():
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    return np.array([(x[i], y[j], z[k]) for i in range(5) for j in range(5) for k in range(4)])


def _init_fcc_like():
    """FCC-like initialization with 100 points."""
    centers = _init_grid()
    np.random.seed(42)
    centers += np.random.normal(0, 0.02, centers.shape)
    return centers


def _init_random(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _init_halton():
    """Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            centers[i, j] = 2 * halton(i + 1, [2, 3, 5][j]) - 1
    return centers


def _fourier_feature_encoding(z, n_freq=8, seed=0):
    """
    Encode a compact latent vector z into 100x3 centers using Fourier features.
    z has dimension 24 (8 per coordinate).
    """
    np.random.seed(seed)
    # Random frequencies and phases for Fourier features
    freqs = np.random.uniform(0.5, 3.0, (n_freq, 3))
    phases = np.random.uniform(0, 2*np.pi, (n_freq, 3))
    
    # z is (24,) -> reshape to (8, 3)
    z = z.reshape(n_freq, 3)
    
    # Compute Fourier features: sum_k z_k * cos(freq_k * t + phase_k)
    # We want to generate 100 points, so t ranges from 0 to 1
    t = np.linspace(0, 1, N)[:, np.newaxis]  # (N, 1)
    
    # For each output dimension (x, y, z):
    # output[d] = sum_k z[k,d] * cos(freq[k,d] * t + phase[k,d])
    # This gives (N, 3)
    
    # Expand: t is (N,1), freqs is (8,3), phases is (8,3), z is (8,3)
    # We need to compute for each n in N, each d in 3:
    # out[n,d] = sum_k z[k,d] * cos(freq[k,d] * t[n] + phase[k,d])
    
    # Reshape for broadcasting
    t_exp = t[:, np.newaxis, :]  # (N, 1, 1)
    freqs_exp = freqs[np.newaxis, :, :]  # (1, 8, 3)
    phases_exp = phases[np.newaxis, :, :]  # (1, 8, 3)
    z_exp = z[np.newaxis, :, :]  # (1, 8, 3)
    
    # Compute: (N, 8, 3)
    angles = freqs_exp * t_exp + phases_exp  # (N, 8, 3)
    cos_vals = np.cos(angles)  # (N, 8, 3)
    
    # out[n,d] = sum_k z[k,d] * cos_vals[n,k,d]
    out = np.sum(z_exp * cos_vals, axis=1)  # (N, 3)
    
    # Scale to fit in [-0.4, 0.4]
    out = out * 0.4
    
    return out


def _fourier_objective_and_grad(z, beta, iu):
    """
    Objective for optimizing latent parameters z of Fourier encoding.
    """
    centers = _fourier_feature_encoding(z, n_freq=8, seed=0)
    x = centers.ravel()
    return _soft_min_objective_and_grad(x, beta, iu)


def _optimize_fourier(deadline, n_latent=24, n_restarts=3):
    """
    Optimize using Fourier feature encoding of latent parameters.
    """
    best_r = -1
    best_centers = None
    iu = np.triu_indices(N, k=1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial latent parameters
        np.random.seed(100 + restart)
        z0 = np.random.uniform(-1, 1, n_latent)
        
        # Get initial centers
        c0 = _fourier_feature_encoding(z0, n_freq=8, seed=0)
        r0 = _compute_radius(c0)
        
        # Optimize latent parameters
        best_z = z0.copy()
        best_r_local = r0
        
        betas = np.geomspace(20, 5000, 5)
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(
                    _fourier_objective_and_grad,
                    best_z,
                    args=(beta, iu),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                new_z = res.x
                new_centers = _fourier_feature_encoding(new_z, n_freq=8, seed=0)
                new_r = _compute_radius(new_centers)
                
                if new_r > best_r_local:
                    best_r_local = new_r
                    best_z = new_z
            except Exception:
                pass
        
        # Refine the best centers from Fourier optimization
        if best_r_local > best_r:
            best_r = best_r_local
            best_centers = _fourier_feature_encoding(best_z, n_freq=8, seed=0).copy()
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Multiple initializations
    inits = [
        _init_grid(),
        _init_fcc_like(),
        _init_random(0),
        _init_halton(),
        _init_random(1),
    ]
    
    best_r = -1
    best_centers = None
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Allocate remaining time divided among remaining inits
        remaining_inits = len(inits) - i
        time_per_init = min(20, (deadline - 15 - time.time()) / max(1, remaining_inits))
        
        # First, try force-directed refinement
        if time.time() < deadline - 10:
            fd_time = min(5, time_per_init / 3)
            fd_centers, fd_r = _force_directed_refinement(init_centers, n_iters=100, deadline=time.time() + fd_time)
            if fd_r > best_r:
                best_r = fd_r
                best_centers = fd_centers.copy()
        
        # Then, try soft-min optimization
        if time.time() < deadline - 10:
            sm_time = min(15, time_per_init / 2)
            sm_centers, sm_r = _optimize_soft_min(fd_centers.ravel() if best_r > 0 else init_centers.ravel(), 
                                                   time.time() + sm_time)
            if sm_r > best_r:
                best_r = sm_r
                best_centers = sm_centers.reshape(N, 3).copy()
    
    # Try Fourier feature encoding optimization if time remains
    if time.time() < deadline - 10:
        ft_time = deadline - 10
        ft_centers, ft_r = _optimize_fourier(ft_time, n_latent=24, n_restarts=2)
        if ft_centers is not None and ft_r > best_r:
            best_r = ft_r
            best_centers = ft_centers.copy()
    
    if best_centers is None:
        best_centers = _init_grid()
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 3):
        best_centers = _init_grid()
    
    return best_centers
