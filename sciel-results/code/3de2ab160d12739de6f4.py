import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # 1. Initialization: Hex Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            c = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    def get_score(c):
        if not np.all(np.isfinite(c)):
            return -np.inf
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        return min(min_pair_dist, min_boundary_dist)
    
    best_c = c.copy()
    best_score = get_score(best_c)
    
    iu, ju = np.triu_indices(N, k=1)
    
    def objective_and_grad(c_flat, beta):
        c = c_flat.reshape(N, 2)
        
        # Pair distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists[iu, ju]
        pair_vals = 0.5 * pair_dists
        
        # Boundary distances
        norms = np.linalg.norm(c, axis=1)
        boundary_vals = 1.0 - norms
        
        all_vals = np.concatenate([pair_vals, boundary_vals])
        
        # Soft min: -1/beta * log(sum(exp(-beta * v)))
        # We want to minimize this.
        # To avoid overflow/underflow, subtract min
        min_val = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - min_val))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0 or not np.isfinite(sum_exp):
             return 0.0, np.zeros_like(c_flat)
             
        log_sum_exp = np.log(sum_exp)
        soft_min_val = min_val - 1.0 / beta * log_sum_exp
        
        # We want to MAXIMIZE soft_min_val.
        # So we MINIMIZE -soft_min_val.
        obj_val = -soft_min_val
        
        # Gradient
        # d/dc soft_min = sum w_k * dv_k/dc
        # d/dc (-soft_min) = - sum w_k * dv_k/dc
        
        weights = exp_vals / sum_exp
        pair_weights = weights[:len(pair_vals)]
        boundary_weights = weights[len(pair_vals):]
        
        grad_soft_min = np.zeros_like(c)
        
        # Boundary grad
        # v_i = 1 - ||c_i||
        # d v_i / d c_i = - c_i / ||c_i||
        norms_safe = np.where(norms > 1e-9, norms, 1.0)
        boundary_grad_vec = -c / norms_safe[:, np.newaxis]
        mask = norms > 1e-9
        boundary_grad_vec[~mask] = 0.0
        
        grad_soft_min += boundary_weights[:, np.newaxis] * boundary_grad_vec
        
        # Pair grad
        # v_ij = 0.5 * ||c_i - c_j||
        # d v_ij / d c_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d v_ij / d c_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        diff_ij = c[iu] - c[ju]
        dist_ij = pair_dists
        dist_ij_safe = np.where(dist_ij > 1e-9, dist_ij, 1.0)
        unit_vec = diff_ij / dist_ij_safe[:, np.newaxis]
        
        contrib = 0.5 * pair_weights[:, np.newaxis] * unit_vec
        
        np.add.at(grad_soft_min, iu, contrib)
        np.add.at(grad_soft_min, ju, -contrib)
        
        grad_obj = -grad_soft_min
        
        if not np.all(np.isfinite(grad_obj)):
             return 0.0, np.zeros_like(c_flat)

        return obj_val, grad_obj.flatten()
    
    # Optimization Loop
    # Phase 1: Moderate beta for global structure
    beta = 50.0
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_and_grad, c.flatten(), args=(beta,), method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.x is not None:
                c_opt = res.x.reshape(N, 2)
                score_opt = get_score(c_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_c = c_opt.copy()
                    c = c_opt
        except Exception:
            pass
    
    # Phase 2: High beta for polishing
    if time.time() < deadline - 2:
        beta = 500.0
        try:
            res = minimize(objective_and_grad, best_c.flatten(), args=(beta,), method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.x is not None:
                c_opt = res.x.reshape(N, 2)
                score_opt = get_score(c_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_c = c_opt.copy()
        except Exception:
            pass
    
    return best_c
