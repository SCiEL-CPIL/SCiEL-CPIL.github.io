import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_softmin(x, beta=100.0):
    """
    Soft-min energy for final polish: approximates the minimum distance.
    x: shape (N, 3)
    Returns (energy, gradient_flat)
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1.0)
    
    d_pairs = np.sqrt(dist_sq_masked)
    d_pairs[~mask] = 1.0
    
    # Soft-min of pair distances
    max_d = np.max(d_pairs)
    exp_terms = np.exp(-beta * (d_pairs - max_d))
    exp_terms[~mask] = 0.0
    
    log_sum = np.log(np.sum(exp_terms))
    soft_min_pair = -1.0 / beta * (log_sum + beta * max_d)
    
    # Gradient
    weights = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)
    weights[~mask] = 0.0
    
    inv_dist = np.where(d_pairs > 1e-12, 1.0 / d_pairs, 0.0)
    inv_dist[~mask] = 0.0
    
    unit_diff = diff * inv_dist[:, :, np.newaxis]
    weighted_unit_diff = unit_diff * weights[:, :, np.newaxis]
    
    grad_soft_pair = (1.0 / beta) * np.sum(weighted_unit_diff, axis=1)
    
    # Wall distances
    d_left = np.maximum(x + 0.5, 1e-10)
    d_right = np.maximum(0.5 - x, 1e-10)
    
    all_wall_d = np.concatenate([d_left.flatten(), d_right.flatten()])
    max_wall = np.max(all_wall_d)
    exp_wall = np.exp(-beta * (all_wall_d - max_wall))
    log_sum_wall = np.log(np.sum(exp_wall))
    soft_min_wall = -1.0 / beta * (log_sum_wall + beta * max_wall)
    
    w_left = np.exp(-beta * (d_left - max_wall)) / np.sum(exp_wall)
    w_right = np.exp(-beta * (d_right - max_wall)) / np.sum(exp_wall)
    
    grad_soft_wall = (1.0 / beta) * (w_left - w_right)
    
    total_energy = -(soft_min_pair + soft_min_wall)
    total_grad = -(grad_soft_pair + grad_soft_wall)
    
    return total_energy, total_grad.flatten()

def _simple_grid():
    """Standard 5x4x5 grid."""
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    return grid[:N]

def _bcc_initial_grid():
    """
    Generate initial points using a BCC-inspired pattern for denser packing.
    """
    points = []
    
    # 5x4x5 grid
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    np.random.seed(42)
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                idx = len(points)
                if idx % 2 == 0:
                    jitter = np.random.uniform(-0.02, 0.02, 3)
                    points.append([x + jitter[0], y + jitter[1], z + jitter[2]])
                else:
                    offset = 0.05
                    points.append([x + offset, y + offset, z + offset])
    
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    bounds = [(-0.5, 0.5)] * 3 * N
    
    # Multiple initial configurations
    init_configs = [
        _simple_grid(),
        _bcc_initial_grid(),
    ]
    
    np.random.seed(123)
    for base in [_simple_grid()]:
        for jitter in [0.03, 0.06, 0.1]:
            init_configs.append(np.clip(base + np.random.uniform(-jitter, jitter, base.shape), -0.49, 0.49))
    
    best_x = init_configs[0].copy()
    best_score = _calc_score(best_x)
    
    # Phase 1: Simulated Annealing
    # SA is structurally different from L-BFGS-B as it allows uphill moves to escape local minima.
    for x0 in init_configs:
        if time.time() > deadline - 40:
            break
            
        x = x0.copy()
        current_score = _calc_score(x)
        
        # SA Parameters
        T_start = 0.05
        T_end = 0.001
        alpha = 0.98 # Cooling rate
        steps = 5000
        step_size = 0.02
        
        T = T_start
        for i in range(steps):
            if time.time() > deadline - 40:
                break
                
            # Random perturbation
            # Move a random subset of points to avoid huge jumps
            n_move = np.random.randint(1, 5)
            indices = np.random.choice(N, n_move, replace=False)
            delta = np.random.normal(0, step_size, (n_move, 3))
            
            x_new = x.copy()
            x_new[indices] += delta
            x_new = np.clip(x_new, -0.5, 0.5)
            
            new_score = _calc_score(x_new)
            
            # Metropolis acceptance criterion
            if new_score > current_score:
                x = x_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                # Accept worse solution with probability exp((new - current) / T)
                # Since we maximize score, delta_score = new - current (negative)
                # P = exp(delta_score / T)
                delta_score = new_score - current_score
                if delta_score < 0:
                    prob = np.exp(delta_score / T)
                    if np.random.rand() < prob:
                        x = x_new
                        current_score = new_score
                
            # Cool down
            T *= alpha
            
            # Adaptive step size: if no improvement in last 100 steps, reduce step size
            # (Simplified: just reduce step size periodically)
            if i % 100 == 0:
                step_size *= 0.95
                
    # Phase 2: Gradient Polish
    # Use the best result from SA (or initial if SA failed) as starting point
    # We will run gradient polish on the global best and a few other good candidates if time permits
    candidates_to_polish = [best_x]
    
    # Add the local best from the last SA run if it's different
    if not np.allclose(best_x, x):
        candidates_to_polish.append(x)
        
    # Also polish the initial grid in case SA went astray (unlikely but safe)
    if time.time() < deadline - 20:
        candidates_to_polish.append(init_configs[0])
        
    for x0 in candidates_to_polish:
        if time.time() > deadline - 10:
            break
            
        x0_flat = x0.flatten()
        
        try:
            # Use a medium beta for stability, then high beta for precision
            betas = [50.0, 200.0, 1000.0]
            
            current_x_flat = x0_flat
            for beta in betas:
                if time.time() > deadline - 5:
                    break
                    
                def obj_soft(x_flat, b=beta):
                    e, g = _energy_softmin(x_flat, beta=b)
                    return e, g
                
                max_iter = 100 if beta < 100 else 200
                res = minimize(obj_soft, current_x_flat, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14})
                
                if res.success:
                    new_x = res.x.reshape(N, 3)
                    new_x = np.clip(new_x, -0.5, 0.5)
                    new_score = _calc_score(new_x)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_x = new_x.copy()
                    
                    current_x_flat = new_x.flatten()
                    
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.clip(best_x, -0.5, 0.5)
    if not np.all(np.isfinite(best_x)):
        best_x = _simple_grid()
    
    return best_x
