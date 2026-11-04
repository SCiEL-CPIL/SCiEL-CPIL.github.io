import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(pos):
    """Compute the maximum radius for a given set of positions."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pos)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist, min_wall_dist) / 2.0

def _soft_min_obj_grad(pos_flat, alpha):
    """
    Smooth approximation of the minimum constraint value.
    We want to maximize min(pairwise_dists, wall_dists).
    Using log-sum-exp: softmin(x) = -log(sum(exp(-alpha*x)))/alpha
    Returns (-softmin, -grad) so we minimize negative softmin.
    """
    pos = pos_flat.reshape(N, 3)
    
    # Pairwise distances (i < j)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Wall distances: 0.5 - |pos_ik| for each point and dim
    wall_dists = 0.5 - np.abs(pos)  # shape (N, 3)
    wall_flat = wall_dists.flatten()
    
    # Combine all constraint values
    all_vals = np.concatenate([d_ij, wall_flat])
    
    # Soft min: -log(sum(exp(-alpha * x))) / alpha
    # For numerical stability, subtract max
    max_val = np.max(all_vals)
    shifted = all_vals - max_val
    exp_neg = np.exp(-alpha * shifted)
    log_sum = np.log(np.sum(exp_neg))
    softmin_val = -(log_sum - alpha * max_val) / alpha
    
    # Gradient of softmin w.r.t. pos
    # d(softmin)/d(x_k) = exp(-alpha*x_k) / sum(exp(-alpha*x_j)) * (-alpha) / alpha
    #                    = -exp(-alpha*x_k) / sum(exp(-alpha*x_j))
    # So gradient of -softmin = exp(-alpha*x_k) / sum(exp(-alpha*x_j))
    
    weights = exp_neg / np.sum(exp_neg)
    
    grad = np.zeros_like(pos_flat)
    
    # Pairwise part
    # d/dpos_i of d_ij = (pos_i - pos_j) / d_ij
    # d/dpos_j of d_ij = (pos_j - pos_i) / d_ij
    # grad of -softmin w.r.t. pos_i += w_ij * (pos_i - pos_j) / d_ij
    # grad of -softmin w.r.t. pos_j += w_ij * (pos_j - pos_i) / d_ij
    
    w_pair = weights[:len(d_ij)]
    d_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
    
    for idx in range(len(d_ij)):
        i = i_idx[idx]
        j = j_idx[idx]
        w = w_pair[idx]
        d = d_safe[idx]
        dir_vec = diff[i, j] / d  # (pos_i - pos_j) / d
        grad[i*3:(i+1)*3] += w * dir_vec
        grad[j*3:(j+1)*3] -= w * dir_vec
    
    # Wall part
    # d/dpos_ik of (0.5 - |pos_ik|) = -sign(pos_ik)
    w_wall = weights[len(d_ij):]
    wall_grad = -np.sign(pos) * w_wall.reshape(N, 3)
    grad += wall_grad.flatten()
    
    return -softmin_val, -grad  # minimize negative softmin

def _optimize_soft_min(pos0, alpha, max_iter=300):
    """Minimize negative soft-min to maximize the minimum constraint."""
    pos0_flat = pos0.flatten()
    
    def objective(x):
        val, _ = _soft_min_obj_grad(x, alpha)
        return val
    
    def gradient(x):
        _, g = _soft_min_obj_grad(x, alpha)
        return g
    
    try:
        res = minimize(objective, pos0_flat, jac=gradient, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 50})
        pos_final = res.x.reshape(N, 3)
        pos_final = np.clip(pos_final, -0.499, 0.499)
        return pos_final
    except Exception:
        return pos0

def _generate_initial_positions(seed_offset=0):
    """Generate diverse initial positions."""
    rng = np.random.RandomState(42 + seed_offset)
    
    # Grid-based with perturbation
    m = int(np.ceil(N ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    pos += rng.normal(0, 0.02, (N, 3))
    pos = np.clip(pos, -0.45, 0.45)
    return pos

def _generate_bcc_positions():
    """BCC-like initial positions for better symmetry."""
    # Use a 5x4x5 grid with BCC offset
    nx, ny, nz = 5, 4, 5
    positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = (ix + 0.5) / nx - 0.5
                y = (iy + 0.5) / ny - 0.5
                z = (iz + 0.5) / nz - 0.5
                positions.append([x, y, z])
                # BCC center
                if ix < nx-1 and iy < ny-1 and iz < nz-1:
                    cx = (ix + 1.0) / nx - 0.5
                    cy = (iy + 1.0) / ny - 0.5
                    cz = (iz + 1.0) / nz - 0.5
                    positions.append([cx, cy, cz])
    
    positions = np.array(positions[:N])
    rng = np.random.RandomState(123)
    positions += rng.normal(0, 0.01, (N, 3))
    positions = np.clip(positions, -0.45, 0.45)
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_pos = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = [
        _generate_initial_positions(0),
        _generate_initial_positions(1),
        _generate_bcc_positions(),
    ]
    
    for init_pos in inits:
        if time.time() > deadline:
            break
        
        pos = init_pos.copy()
        current_r = _compute_radius(pos)
        
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Anneal alpha: start large (sharp min), decrease for smoother optimization
        alphas = [50, 30, 20, 15, 10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1.2, 1.0]
        
        for alpha in alphas:
            if time.time() > deadline:
                break
            
            pos = _optimize_soft_min(pos, alpha, max_iter=200)
            pos = np.clip(pos, -0.499, 0.499)
            
            new_r = _compute_radius(pos)
            if new_r > current_r:
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
        
        # Final polish with very fine alpha
        for _ in range(3):
            if time.time() > deadline:
                break
            pos = _optimize_soft_min(pos, 0.8, max_iter=100)
            pos = np.clip(pos, -0.499, 0.499)
            new_r = _compute_radius(pos)
            if new_r > current_r:
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
            else:
                break
    
    if best_pos is None:
        # Fallback
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pos = grid[:N].copy()
        best_pos += np.random.normal(0, 0.01, (N, 3))
        best_pos = np.clip(best_pos, -0.49, 0.49)
    
    return best_pos
