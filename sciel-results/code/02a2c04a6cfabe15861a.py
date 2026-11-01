"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of squared distances.
    This is a smooth approximation of -min(d_i^2).
    """
    c = params.reshape(N, 2)
    
    # Pair distances
    dx = c[:, 0, None] - c[:, None, 0]
    dy = c[:, 1, None] - c[:, None, 1]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1e10)  # Large value to ignore diagonal
    
    # Boundary distances
    d_left = 0.5 - c[:, 0]
    d_right = 0.5 + c[:, 0]
    d_down = 0.5 - c[:, 1]
    d_up = 0.5 + c[:, 1]
    bd2 = np.concatenate([d_left**2, d_right**2, d_down**2, d_up**2])
    
    # Combine all squared distances
    all_d2 = np.concatenate([d2.ravel(), bd2])
    
    # Log-sum-exp for numerical stability
    max_d2 = np.max(all_d2)
    log_sum_exp = max_d2 + np.log(np.sum(np.exp(all_d2 - max_d2)))
    
    # Negative because we want to maximize the minimum distance
    E = -log_sum_exp / beta
    
    # Gradient
    exp_term = np.exp(all_d2 - max_d2)
    weights = exp_term / np.sum(exp_term)
    
    grad = np.zeros_like(params)
    
    # Pair gradient
    pair_weights = weights[:N*N].reshape(N, N)
    np.fill_diagonal(pair_weights, 0)
    
    dx_grad = (2 * dx * pair_weights).sum(axis=1)
    dy_grad = (2 * dy * pair_weights).sum(axis=1)
    
    # Boundary gradient
    b_weights = weights[N*N:]
    d_left_grad = 2 * d_left * b_weights[:N]
    d_right_grad = -2 * d_right * b_weights[N:2*N]
    d_down_grad = 2 * d_down * b_weights[2*N:3*N]
    d_up_grad = -2 * d_up * b_weights[3*N:]
    
    grad[0::2] = dx_grad + d_left_grad + d_right_grad
    grad[1::2] = dy_grad + d_down_grad + d_up_grad
    
    grad *= -1.0 / beta
    
    return E, grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try multiple spacings
    for d in np.linspace(0.055, 0.085, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.495, 0.495)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, time_budget=20.0):
    """Run L-BFGS-B energy minimization with tight tolerances."""
    start = time.time()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        res = minimize(
            _soft_min_energy,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
        r_opt = _compute_radius(c_opt)
        elapsed = time.time() - start
        return c_opt, r_opt, elapsed
    except Exception:
        return c0, _compute_radius(c0), time.time() - start

def _local_polish(c, time_limit=30.0):
    """
    Multi-stage local polish: coarse-to-fine step sizes.
    Stage 1: step ~1e-3, identify and resolve conflicts
    Stage 2: step ~1e-4, refine
    Stage 3: step ~1e-5, high-precision
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    stages = [
        (1e-3, 0.4),   # (step, fraction of time)
        (3e-4, 0.3),
        (1e-4, 0.2),
        (3e-5, 0.1),
    ]
    
    for step, frac in stages:
        stage_deadline = start_time + time_limit * frac
        while time.time() < stage_deadline:
            if time.time() - start_time > time_limit:
                break
            
            # Compute full distance matrix
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair_dist = np.min(dist_matrix)
            
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            min_boundary_dist = np.min(bdists)
            
            current_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            tol = 1e-7
            forces = np.zeros((N, 2))
            
            # Boundary forces
            d_left = 0.5 - c[:, 0]
            d_right = c[:, 0] + 0.5
            d_down = 0.5 - c[:, 1]
            d_up = c[:, 1] + 0.5
            min_d_vec = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
            
            tight_b = np.where(min_d_vec < min_boundary_dist + tol)[0]
            for i in tight_b:
                if d_left[i] == min_d_vec[i]:
                    forces[i, 0] += 1.0
                elif d_right[i] == min_d_vec[i]:
                    forces[i, 0] -= 1.0
                elif d_down[i] == min_d_vec[i]:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
            
            # Pair forces
            tight_pairs = np.where(dist_matrix < (min_pair_dist + tol))
            i_idx = tight_pairs[0]
            j_idx = tight_pairs[1]
            mask = i_idx < j_idx
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            
            if len(i_idx) > 0:
                vecs = c[i_idx] - c[j_idx]
                norms = np.linalg.norm(vecs, axis=1)
                safe_norms = np.where(norms > 1e-10, norms, 1.0)
                unit_vecs = vecs / safe_norms[:, np.newaxis]
                np.add.at(forces, i_idx, unit_vecs)
                np.add.at(forces, j_idx, -unit_vecs)
            
            # Apply forces
            improved = False
            for i in range(N):
                norm_f = np.linalg.norm(forces[i])
                if norm_f > 1e-10:
                    new_c_i = c[i] + (forces[i] / norm_f) * step
                    if np.all(np.abs(new_c_i) < 0.499):
                        old_c_i = c[i].copy()
                        c[i] = new_c_i
                        new_r = _compute_radius(c)
                        if new_r > current_r:
                            current_r = new_r
                            improved = True
                        else:
                            c[i] = old_c_i
            
            if not improved:
                break
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_c = None
    best_r = 0.0
    
    # Generate diverse initializations
    inits = [_initialize_hexagonal()]
    
    # Add a jittered hexagonal variant
    hex_c = _initialize_hexagonal()
    rng = np.random.default_rng(123)
    jittered = hex_c + rng.uniform(-0.005, 0.005, (N, 2))
    jittered = np.clip(jittered, -0.495, 0.495)
    inits.append(jittered)
    
    for c0 in inits:
        if time.time() > deadline - 60:
            break
        
        # Energy optimization
        c_opt, r_opt, elapsed = _optimize_energy(c0, time_budget=25.0)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Local polish
        remaining = deadline - time.time() - 20
        if remaining > 5 and best_c is not None:
            c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 40.0))
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
    
    # Final safety check
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
