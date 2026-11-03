"""Improved sphere packing: 50 equal spheres in unit cube."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _compute_radius(c):
    """Compute exact max radius for given centers."""
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    wall = 0.5 - np.max(np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """Soft-min objective and gradient."""
    c = x.reshape(N, 3)
    
    # Wall clearances
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    # Pairwise distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]
    
    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)
    
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta
    
    weights = exp_vals / sum_exp
    
    grad_x = np.zeros_like(c)
    
    wall_weights = weights[:3*N].reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= wall_weights * sign_c
    
    pair_weights = weights[3*N:]
    
    ci = c[i]
    cj = c[j]
    diff = ci - cj
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    
    grad_ci = diff / (2.0 * norms)
    grad_cj = -grad_ci
    
    weighted_grad_ci = pair_weights[:, np.newaxis] * grad_ci
    weighted_grad_cj = pair_weights[:, np.newaxis] * grad_cj
    
    np.add.at(grad_x, i, weighted_grad_ci)
    np.add.at(grad_x, j, weighted_grad_cj)
    
    f = -soft_min_val
    g = -grad_x.ravel()
    
    return f, g

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    rng = np.random.default_rng(42)
    
    # BCC lattice
    n_div = 5
    bcc_points = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                bcc_points.append([i / n_div, j / n_div, k / n_div])
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                bcc_points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    bcc_points = np.array(bcc_points) - 0.5
    
    # Select from BCC
    idx = np.arange(len(bcc_points))
    rng.shuffle(idx)
    selected = [bcc_points[idx[0]]]
    for i in range(1, len(idx)):
        if len(selected) >= N:
            break
        p = bcc_points[idx[i]]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    while len(selected) < N:
        selected.append(rng.uniform(-0.45, 0.45, size=3))
    configs.append(np.array(selected[:N]))
    
    # FCC lattice
    n_div = 4
    fcc_points = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                fcc_points.append([i / n_div, j / n_div, k / n_div])
    for i in range(n_div):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                fcc_points.append([(i + 0.5) / n_div, j / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div):
            for k in range(n_div + 1):
                fcc_points.append([i / n_div, (j + 0.5) / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div):
                fcc_points.append([i / n_div, j / n_div, (k + 0.5) / n_div])
    fcc_points = np.array(fcc_points) - 0.5
    mask = np.all(np.abs(fcc_points) <= 0.5, axis=1)
    fcc_points = fcc_points[mask]
    
    idx = np.arange(len(fcc_points))
    rng.shuffle(idx)
    selected = [fcc_points[idx[0]]]
    for i in range(1, len(idx)):
        if len(selected) >= N:
            break
        p = fcc_points[idx[i]]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.09:
            selected.append(p)
    while len(selected) < N:
        selected.append(rng.uniform(-0.45, 0.45, size=3))
    configs.append(np.array(selected[:N]))
    
    # Grid-based
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.arange(len(grid))
    rng.shuffle(idx)
    grid_shuffled = grid[idx]
    
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)
    while len(selected) < N:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    configs.append(np.array(selected[:N]))
    
    # Random with repulsion
    for trial in range(3):
        c = rng.uniform(-0.4, 0.4, size=(N, 3))
        # Simple repulsion refinement
        for _ in range(100):
            d = cdist(c, c)
            np.fill_diagonal(d, np.inf)
            for i in range(N):
                force = np.zeros(3)
                for j in range(N):
                    if i != j:
                        diff = c[i] - c[j]
                        dist = np.linalg.norm(diff)
                        if dist < 0.15:
                            force += diff / (dist ** 2 + 1e-10) * 0.01
                # Wall repulsion
                for k in range(3):
                    if c[i, k] > 0.35:
                        force[k] -= 0.01
                    elif c[i, k] < -0.35:
                        force[k] += 0.01
                c[i] += force
                c[i] = np.clip(c[i], -0.45, 0.45)
        configs.append(c.copy())
    
    # Perturbed BCC
    c_bcc = configs[0].copy()
    for seed in range(4):
        noise = rng.normal(0, 0.015, size=(N, 3))
        c_variant = c_bcc + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        configs.append(c_variant)
    
    # Perturbed FCC
    c_fcc = configs[1].copy()
    for seed in range(3):
        noise = rng.normal(0, 0.012, size=(N, 3))
        c_variant = c_fcc + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        configs.append(c_variant)
    
    return configs

def _optimize_from_initial(c_init, deadline):
    """Run continuation optimization from an initial configuration."""
    best_c = c_init.copy()
    best_val = _compute_radius(best_c)
    
    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        def obj_grad(x):
            return _soft_min_objective_and_grad(x, beta)
        
        try:
            res = minimize(obj_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4995, 0.4995)
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # High-precision polish
    if time.time() < deadline - 10:
        for beta_polish in [5000, 10000, 20000]:
            if time.time() > deadline - 5:
                break
            
            def obj_grad_polish(x):
                return _soft_min_objective_and_grad(x, beta_polish)
            
            try:
                res = minimize(obj_grad_polish, best_c.ravel(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxls': 30})
                
                new_c = res.x.reshape(N, 3)
                new_c = np.clip(new_c, -0.49999, 0.49999)
                new_val = _compute_radius(new_c)
                
                if new_val > best_val:
                    best_val = new_val
                    best_c = new_c.copy()
                    
            except Exception:
                pass
    
    # Coordinate-wise refinement
    if time.time() < deadline - 2:
        step_sizes = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
        for step in step_sizes:
            if time.time() > deadline - 0.5:
                break
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in range(N):
                    for k in range(3):
                        for sign in [-1, 1]:
                            c_trial = best_c.copy()
                            c_trial[i, k] += sign * step
                            c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                            val_trial = _compute_radius(c_trial)
                            if val_trial > best_val + 1e-12:
                                best_val = val_trial
                                best_c = c_trial
                                improved = True
    
    return best_c, best_val

def _active_set_refine(c, deadline):
    """Refine using only active constraints (binding ones)."""
    best_c = c.copy()
    best_val = _compute_radius(best_c)
    
    if time.time() > deadline - 5:
        return best_c, best_val
    
    # Find active constraints
    wall = 0.5 - np.abs(best_c)
    d = cdist(best_c, best_c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    r = min(wall, min_pair)
    
    # Active wall constraints
    active_wall = np.where(wall <= r + 1e-6)
    # Active pair constraints
    active_pairs = np.where(d / 2.0 <= r + 1e-6)
    active_pairs = [(i, j) for i, j in zip(active_pairs[0], active_pairs[1]) if i < j]
    
    # Try to push active constraints outward
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        
        improved = False
        # Try moving spheres away from walls
        for idx in active_wall:
            i, k = idx
            for sign in [-1, 1]:
                c_trial = best_c.copy()
                c_trial[i, k] -= sign * 1e-6
                c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                val_trial = _compute_radius(c_trial)
                if val_trial > best_val + 1e-12:
                    best_val = val_trial
                    best_c = c_trial
                    improved = True
        
        # Try separating close pairs
        for i, j in active_pairs[:20]:  # Limit to 20 most active pairs
            diff = best_c[i] - best_c[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                c_trial = best_c.copy()
                c_trial[i] += direction * 1e-6
                c_trial[j] -= direction * 1e-6
                c_trial = np.clip(c_trial, -0.499999, 0.499999)
                val_trial = _compute_radius(c_trial)
                if val_trial > best_val + 1e-12:
                    best_val = val_trial
                    best_c = c_trial
                    improved = True
        
        if not improved:
            break
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    configs = _generate_initial_configs()
    
    best_c = configs[0].copy()
    best_val = _compute_radius(best_c)
    
    # Optimize from each initial configuration
    for idx, c_init in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        try:
            c_opt, val = _optimize_from_initial(c_init, deadline)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Active set refinement on best result
    if time.time() < deadline - 5:
        best_c, best_val = _active_set_refine(best_c, deadline)
    
    # Final coordinate-wise polish
    if time.time() < deadline - 1:
        step_sizes = [1e-6, 5e-7, 1e-7, 5e-8]
        for step in step_sizes:
            if time.time() > deadline - 0.5:
                break
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in range(N):
                    for k in range(3):
                        for sign in [-1, 1]:
                            c_trial = best_c.copy()
                            c_trial[i, k] += sign * step
                            c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                            val_trial = _compute_radius(c_trial)
                            if val_trial > best_val + 1e-12:
                                best_val = val_trial
                                best_c = c_trial
                                improved = True
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c
