"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with diverse hexagonal lattices (rotations, offsets).
2. Global optimization with adaptive beta scheduling.
3. Multi-start local polish with constraint-aware refinement.
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
    Smooth approximation of -log(sum(exp(-beta * d_i^2))).
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d2_pairs = d2[iu]
    
    # Log-Sum-Exp with shift for stability
    d2_shifted = d2_pairs - np.min(d2_pairs)
    weights = np.exp(-beta * d2_shifted)
    p = weights / np.sum(weights)
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    grad_x[i_idx] -= 2.0 * p * vec_x
    grad_x[j_idx] += 2.0 * p * vec_x
    grad_y[i_idx] -= 2.0 * p * vec_y
    grad_y[j_idx] += 2.0 * p * vec_y
    
    # Wall penalty
    eps = 1e-6
    dx_w_left = np.clip(0.5 - x, eps, None)
    dx_w_right = np.clip(0.5 + x, eps, None)
    dy_w_down = np.clip(0.5 - y, eps, None)
    dy_w_up = np.clip(0.5 + y, eps, None)
    
    E_wall = (1.0/dx_w_left**2 + 1.0/dx_w_right**2 + 1.0/dy_w_down**2 + 1.0/dy_w_up**2).sum()
    
    E_total = E_wall + 1.0 * (np.log(np.sum(weights)) / beta)
    
    grad_x += -2.0 / dx_w_left**3 + 2.0 / dx_w_right**3
    grad_y += -2.0 / dy_w_down**3 + 2.0 / dy_w_up**3
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad

def _initialize_hexagonal():
    """Initialize with diverse hexagonal patterns."""
    best_c = None
    best_r = 0
    
    # Try multiple spacings, rotations, and offsets
    for d in np.linspace(0.055, 0.085, 15):
        for rot_angle in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            for x_off in [0, 0.01, -0.01, 0.02]:
                for y_off in [0, 0.01, -0.01, 0.02]:
                    points = []
                    row_spacing = d * np.sqrt(3) / 2
                    x_offset = d / 2
                    
                    y = -0.5 + d / 2 + y_off
                    row = 0
                    while y <= 0.5 - d / 2:
                        x = -0.5 + d / 2 + x_off
                        if row % 2 == 1:
                            x += x_offset
                        while x <= 0.5 - d / 2:
                            # Apply rotation
                            xr = x * np.cos(rot_angle) - y * np.sin(rot_angle)
                            yr = x * np.sin(rot_angle) + y * np.cos(rot_angle)
                            points.append([xr, yr])
                            x += d
                        y += row_spacing
                        row += 1
                    
                    if len(points) < N:
                        continue
                    
                    points = np.array(points)
                    # Keep only points inside the square
                    mask = (np.abs(points[:, 0]) <= 0.5 - d/2) & (np.abs(points[:, 1]) <= 0.5 - d/2)
                    points = points[mask]
                    
                    if len(points) < N:
                        continue
                    
                    c = points[:N].copy()
                    c = np.clip(c, -0.495, 0.495)
                    r = _compute_radius(c)
                    if r > best_r:
                        best_r = r
                        best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, beta=100.0, maxiter=150):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        res = minimize(
            lambda p: _soft_min_energy(p, beta=beta),
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
        return c_opt
    except Exception:
        return c0

def _local_polish(c, time_limit=30.0, beta=300.0):
    """
    Fast local polish using constraint-aware gradient refinement.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    def _polish_energy(params, beta=300.0):
        """Energy focusing on active constraints."""
        c_ = params.reshape(N, 2)
        x = c_[:, 0]
        y = c_[:, 1]
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        np.fill_diagonal(d2, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d2_pairs = d2[iu]
        
        # Boundary distances
        bdists = np.minimum(
            np.minimum(0.5 - x, 0.5 + x),
            np.minimum(0.5 - y, 0.5 + y)
        )
        
        # Log-sum-exp for pairs
        d2_shifted = d2_pairs - np.min(d2_pairs)
        w_pairs = np.exp(-beta * d2_shifted)
        E_pairs = np.log(np.sum(w_pairs)) / beta
        
        # Log-sum-exp for boundaries
        bd_shifted = bdists - np.min(bdists)
        w_bounds = np.exp(-beta * bd_shifted)
        E_bounds = np.log(np.sum(w_bounds)) / beta
        
        E = E_pairs + E_bounds
        
        # Gradients
        p_pairs = w_pairs / np.sum(w_pairs)
        p_bounds = w_bounds / np.sum(w_bounds)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        
        grad_x[i_idx] -= 2.0 * p_pairs * vec_x
        grad_x[j_idx] += 2.0 * p_pairs * vec_x
        grad_y[i_idx] -= 2.0 * p_pairs * vec_y
        grad_y[j_idx] += 2.0 * p_pairs * vec_y
        
        # Boundary gradient
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        
        d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
        d_min = np.min(d_all, axis=1, keepdims=True)
        wall_weights = np.exp(-beta * (d_all - d_min))
        wall_weights /= wall_weights.sum(axis=1, keepdims=True)
        
        grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
        grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
        
        total_grad = np.concatenate([grad_x, grad_y])
        return E, total_grad
    
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    try:
        res = minimize(
            lambda p: _polish_energy(p, beta=beta),
            c.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.4999, 0.4999)
        r_opt = _compute_radius(c_opt)
        if r_opt > current_r:
            c = c_opt
            current_r = r_opt
    except Exception:
        pass
    
    # Micro-adjustment: nudge active points
    if time.time() - start_time < time_limit:
        for _ in range(30):
            if time.time() - start_time > time_limit - 1:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            
            min_pair_dist = np.min(dist_matrix)
            min_boundary_dist = np.min(bdists)
            target_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if target_r > current_r + 1e-12:
                current_r = target_r
            
            tol = 1e-8
            active_pairs = np.argwhere(dist_matrix < min_pair_dist + tol)
            active_bounds = np.where(bdists < min_boundary_dist + tol)[0]
            
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                break
            
            forces = np.zeros((N, 2))
            
            for i, j in active_pairs:
                if i == j:
                    continue
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    unit_vec = vec / norm
                    forces[i] += unit_vec
                    forces[j] -= unit_vec
            
            for i in active_bounds:
                d_left = 0.5 - c[i, 0]
                d_right = c[i, 0] + 0.5
                d_down = 0.5 - c[i, 1]
                d_up = c[i, 1] + 0.5
                
                min_d = min(d_left, d_right, d_down, d_up)
                if d_left == min_d:
                    forces[i, 0] += 1.0
                elif d_right == min_d:
                    forces[i, 0] -= 1.0
                elif d_down == min_d:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
            
            moved = False
            for step in [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
                if moved:
                    break
                for i in range(N):
                    if np.linalg.norm(forces[i]) > 1e-10:
                        direction = forces[i] / np.linalg.norm(forces[i])
                        new_pos = c[i] + direction * step
                        if np.all(np.abs(new_pos) < 0.49999):
                            old_pos = c[i].copy()
                            c[i] = new_pos
                            new_r = _compute_radius(c)
                            if new_r > current_r:
                                current_r = new_r
                                moved = True
                            else:
                                c[i] = old_pos
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Phase 1: Diverse initialization
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Phase 2: Global optimization with adaptive beta
    beta_schedule = [20.0, 40.0, 70.0, 100.0]
    for beta in beta_schedule:
        if time.time() > deadline - 80:
            break
        c_opt = _optimize_energy(best_c, beta=beta, maxiter=120)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Phase 3: Multi-start local polish
    remaining = deadline - time.time() - 5
    if remaining > 15:
        # Try polishing from the current best
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining * 0.4, 40.0), beta=300.0)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
        
        # Try polishing from slightly perturbed versions
        remaining = deadline - time.time() - 5
        if remaining > 10:
            for trial in range(3):
                if time.time() > deadline - 5:
                    break
                # Add small random perturbation
                perturbation = np.random.randn(N, 2) * 0.001
                c_perturbed = best_c + perturbation
                c_perturbed = np.clip(c_perturbed, -0.499, 0.499)
                
                c_polished2, r_polished2 = _local_polish(c_perturbed, time_limit=min(remaining * 0.3, 25.0), beta=400.0)
                if r_polished2 > best_r:
                    best_r = r_polished2
                    best_c = c_polished2.copy()
    
    # Phase 4: Final high-precision polish
    remaining = deadline - time.time() - 2
    if remaining > 5:
        c_final, r_final = _local_polish(best_c, time_limit=min(remaining, 20.0), beta=500.0)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
