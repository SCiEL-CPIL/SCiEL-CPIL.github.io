import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_inverse_sq(x):
    """
    Inverse-square energy: strong repulsion when spheres are close.
    x: shape (N, 3)
    Returns (energy, gradient_flat)
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    
    # Energy: sum of 1/d^2 for all pairs
    e_pairs = np.sum(1.0 / dist_sq_safe)
    
    # Gradient of sum(1/d^2) w.r.t. x_i:
    # d/dx_i [1/|x_i - x_j|^2] = -2*(x_i - x_j)/|x_i - x_j|^4
    factor = 1.0 / (dist_sq_safe ** 2)
    factor[~mask] = 0.0
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad_pairs = -2.0 * np.sum(weighted_diff, axis=1)
    
    # Wall distances
    d_left = np.maximum(x + 0.5, 1e-8)
    d_right = np.maximum(0.5 - x, 1e-8)
    
    e_walls = np.sum(1.0 / (d_left ** 2)) + np.sum(1.0 / (d_right ** 2))
    
    # Gradient of wall terms:
    # d/dx_ik [1/(x_ik+0.5)^2] = -2/(x_ik+0.5)^3
    # d/dx_ik [1/(0.5-x_ik)^2] = +2/(0.5-x_ik)^3
    grad_walls = -2.0 / (d_left ** 3) + 2.0 / (d_right ** 3)
    
    total_energy = e_pairs + e_walls
    total_grad = grad_pairs + grad_walls
    
    return total_energy, total_grad.flatten()


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


def _bcc_initial_grid():
    """
    Generate initial points using a BCC-inspired pattern for denser packing.
    Combines a regular grid with body-center offsets.
    """
    points = []
    
    # 5x4x5 grid
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    
    # Add small BCC-like perturbations to break symmetry
    np.random.seed(42)
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                # Alternate between corner and body-center positions
                idx = len(points)
                if idx % 2 == 0:
                    # Corner position with small jitter
                    jitter = np.random.uniform(-0.02, 0.02, 3)
                    points.append([x + jitter[0], y + jitter[1], z + jitter[2]])
                else:
                    # Body-center offset
                    offset = 0.05
                    points.append([x + offset, y + offset, z + offset])
    
    return np.array(points[:N])


def _simple_grid():
    """Standard 5x4x5 grid."""
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    return grid[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    bounds = [(-0.5, 0.5)] * 3 * N
    
    # Multiple initial configurations
    init_configs = [
        _simple_grid(),
        _bcc_initial_grid(),
    ]
    
    # Add jittered versions
    np.random.seed(123)
    for base in [_simple_grid()]:
        for jitter in [0.03, 0.06, 0.1]:
            init_configs.append(np.clip(base + np.random.uniform(-jitter, jitter, base.shape), -0.49, 0.49))
    
    best_x = init_configs[0].copy()
    best_score = _calc_score(best_x)
    
    # Phase 1: Inverse-square energy optimization (strong repulsion)
    for x0 in init_configs:
        if time.time() > deadline - 30:
            break
        
        x0_flat = x0.flatten()
        
        try:
            def obj_inv_sq(x_flat):
                e, g = _energy_inverse_sq(x_flat)
                return e, g
            
            res = minimize(obj_inv_sq, x0_flat, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success:
                new_x = res.x.reshape(N, 3)
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Soft-min optimization with increasing beta (sharpen minimum)
    if time.time() < deadline - 20:
        current_x = best_x.flatten()
        
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj_soft(x_flat, b=beta):
                    e, g = _energy_softmin(x_flat, beta=b)
                    return e, g
                
                max_iter = 100 if beta < 100 else 200
                res = minimize(obj_soft, current_x, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14})
                
                if res.success:
                    new_x = res.x.reshape(N, 3)
                    new_x = np.clip(new_x, -0.5, 0.5)
                    new_score = _calc_score(new_x)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_x = new_x.copy()
                    
                    current_x = new_x.flatten()
                    
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            def obj_final(x_flat):
                e, g = _energy_softmin(x_flat, beta=1000.0)
                return e, g
            
            res_final = minimize(obj_final, best_x.flatten(), method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if res_final.success:
                final_x = res_final.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x.copy()
                    
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.clip(best_x, -0.5, 0.5)
    if not np.all(np.isfinite(best_x)):
        best_x = _simple_grid()
    
    return best_x
