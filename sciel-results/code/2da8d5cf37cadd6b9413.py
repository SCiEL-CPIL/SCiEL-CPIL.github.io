import numpy as np
import time
import math
from scipy.optimize import least_squares, minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def polish_least_squares(points, time_budget, k=100.0, R_target=0.06):
    """
    Polish using scipy.optimize.least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container to ensure validity during optimization
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        # Use trf (trust-region reflective)
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=20000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        return points, compute_score(points)

def polish_lbfgsb(points, time_budget, k=100.0):
    """
    Polish using scipy.optimize.minimize with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        # We want to maximize s_val, so minimize -s_val
        return -s_val
    
    x0 = points.flatten()
    
    try:
        # Use L-BFGS-B with bounds to keep points roughly inside
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12}
        )
        
        x_opt = result.x.reshape(N, 2)
        
        # Project inside
        rad = np.linalg.norm(x_opt, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_opt[mask] /= rad[mask, np.newaxis]
        
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        return points, compute_score(points)

def polish_nn(latent, W, b, time_budget, k=100.0):
    """
    Polish using a neural network layer to break symmetry.
    latent: (M, 2) fixed latent variables
    W: (M, 2N) weights
    b: (2N,) bias
    """
    start_time = time.time()
    M = latent.shape[0]
    
    def decode(params):
        # params: flat array of W and b
        W_flat = params[:M * 2 * N]
        b_flat = params[M * 2 * N:]
        W = W_flat.reshape(M, 2, N).transpose(0, 2, 1) # (M, N, 2)
        b = b_flat.reshape(N, 2)
        
        # Compute output: sum_m latent[m] * W[m, :, :] + b
        # latent: (M, 2), W: (M, N, 2)
        # out[i, j] = sum_m latent[m, j] * W[m, i, j] + b[i, j]
        # Using einsum: 'mj,mij->ij'
        out = np.einsum('mj,mij->ij', latent, W) + b
        return out
    
    def objective(params):
        x = decode(params)
        # Project inside container
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return -s_val
    
    # Initialize params from current W and b
    x0 = np.concatenate([W.flatten(), b.flatten()])
    
    try:
        # Bounds for W and b to prevent explosion
        # W magnitude shouldn't be too large, b shouldn't be too large
        bounds = [(-10.0, 10.0)] * len(x0)
        
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12}
        )
        
        params_opt = result.x
        x_opt = decode(params_opt)
        
        # Project inside
        rad = np.linalg.norm(x_opt, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_opt[mask] /= rad[mask, np.newaxis]
        
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        # Fallback to decoding the initial params
        x_fallback = decode(x0)
        rad = np.linalg.norm(x_fallback, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x_fallback[mask] /= rad[mask, np.newaxis]
        return x_fallback, compute_score(x_fallback)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Optimization Phase
    
    # First: L-BFGS-B polish
    lbfgsb_budget = min(20.0, time_limit - (time.time() - start_time) - 5.0)
    if lbfgsb_budget > 5.0:
        polished_points, polished_score = polish_lbfgsb(best_points, lbfgsb_budget)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
    
    # Second: Neural Network Polish
    # Set up NN: M latent variables
    M = 20
    latent = np.random.normal(0, 1, size=(M, 2))
    
    # Initialize W such that output is close to best_points
    # We want: sum_m latent[m] * W[m, i, :] + b[i, :] ~ best_points[i, :]
    # Simple init: W = 0, b = best_points
    # But this doesn't use the latent variables effectively.
    # Better: Solve for W, b approximately? Or just random small W?
    # Let's try W=0, b=best_points first, then let optimizer move it.
    # Actually, if W=0, the gradient wrt W is latent * dL/dx, which is non-zero.
    # So it will start moving.
    
    W = np.zeros((M, N, 2))
    b = best_points.copy()
    
    nn_budget = min(30.0, time_limit - (time.time() - start_time) - 5.0)
    if nn_budget > 5.0:
        polished_points, polished_score = polish_nn(latent, W, b, nn_budget)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
    
    # Third: Run multiple least_squares polishes with perturbations
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    num_ls_runs = 3
    if remaining > 15.0:
        time_per_run = remaining / (num_ls_runs + 1)
        
        for i in range(num_ls_runs):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            if i == 0:
                current_points = best_points.copy()
            else:
                noise_scale = 0.015 * (1 + i * 0.3)
                noise = np.random.normal(0, noise_scale, size=best_points.shape)
                current_points = best_points + noise
                rad = np.linalg.norm(current_points, axis=1)
                mask = rad > 1.0
                if np.any(mask):
                    current_points[mask] /= rad[mask, np.newaxis]
                    
            run_time_budget = min(time_per_run, time_limit - (time.time() - start_time) - 2.0)
            
            if run_time_budget < 1.0:
                continue
            
            polished_points, polished_score = polish_least_squares(current_points, run_time_budget)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points
    
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
