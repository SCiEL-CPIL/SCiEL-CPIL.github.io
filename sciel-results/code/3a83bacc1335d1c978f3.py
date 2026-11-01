import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    # Generate hex lattice points
    # Spacing s such that we get at least n points in unit circle
    # Approx area per point s^2 * sqrt(3)/2. Total area pi.
    # n * s^2 * sqrt(3)/2 <= pi -> s <= sqrt(2*pi / (n*sqrt(3)))
    # We want to pack them in, so we start with a slightly tighter lattice
    # and let the optimizer expand it or adjust positions.
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.98 # Tighter margin
    
    # Generate grid
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
                
    points = np.array(points)
    # Sort by distance to center
    points = points[points[:, 2].argsort()]
    
    # If we have more than N, take the first N
    if len(points) >= n:
        return points[:n, :2]
    else:
        # Fallback: random or scaled
        r_max = np.max(points[:, 2])
        scale = 1.0 / r_max
        return points[:, :2] * scale

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    x0 = _hex_lattice_init(N)
    
    # Precompute indices for pairs
    idx_i, idx_j = np.triu_indices(N, 1)
    M = len(idx_i)
    
    def calc_score(x):
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        min_d_ij = np.min(d_ij) / 2.0
        
        r_i = np.linalg.norm(x, axis=1)
        min_d_bound = np.min(1.0 - r_i)
        
        return np.min([min_d_ij, min_d_bound])
    
    def make_objective(beta):
        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, 2)
            # Calculate distances
            diff = x[idx_i] - x[idx_j]
            d_ij = np.linalg.norm(diff, axis=1)
            
            # Avoid division by zero
            d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
            
            # Boundary distances
            r_i = np.linalg.norm(x, axis=1)
            d_bound = 1.0 - r_i
            
            # Combine all constraints y
            # y_k = d_ij / 2
            # y_m = d_bound
            y_pairs = d_ij / 2.0
            y_bound = d_bound
            all_y = np.concatenate([y_pairs, y_bound])
            
            # Soft min
            # softmin(y) = -1/beta * log(sum(exp(-beta*y)))
            # We want to maximize softmin, so minimize -softmin.
            # -softmin = 1/beta * log(sum(exp(-beta*y)))
            
            # Shift for numerical stability
            y_min = np.min(all_y)
            exp_terms = np.exp(-beta * (all_y - y_min))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Objective value
            obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
            
            # Gradient
            # grad(-softmin) = - sum w_k * grad(y_k)
            
            grad = np.zeros_like(x)
            
            # Gradient of y_pairs
            # y = d/2. grad_c_i (d/2) = (c_i - c_j) / (2 * d)
            # grad_c_j (d/2) = (c_j - c_i) / (2 * d) = - grad_c_i
            w_pairs = weights[:M]
            
            terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
            
            # Accumulate to i: -terms
            np.add.at(grad, idx_i, -terms)
            # Accumulate to j: +terms
            np.add.at(grad, idx_j, terms)
            
            # Gradient of y_bound
            # y = 1 - |c|. grad_c (1 - |c|) = - c / |c|
            w_bound = weights[M:]
            
            r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
            # grad(-softmin)_bound_part = w_bound * c_i/r_i
            grad += (w_bound[:, np.newaxis] * (x / r_i_safe[:, np.newaxis]))
            
            return obj_val, grad.flatten()
        return objective_and_grad

    best_x = x0
    best_score = calc_score(best_x)
    
    # Define optimization stages
    # (beta, maxiter, gtol, ftol)
    stages = [
        (20.0, 100, 1e-5, 1e-5),
        (50.0, 100, 1e-6, 1e-6),
        (100.0, 100, 1e-7, 1e-7),
        (200.0, 100, 1e-8, 1e-8),
        (500.0, 50, 1e-9, 1e-9),
        (1000.0, 20, 1e-10, 1e-10)
    ]
    
    for beta, maxiter, gtol, ftol in stages:
        if time.time() - start_time > time_limit - 10:
            break
            
        obj = make_objective(beta)
        try:
            res = minimize(
                obj, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol}
            )
            candidate = res.x.reshape(N, 2)
            s = calc_score(candidate)
            if s > best_score:
                best_x = candidate
                best_score = s
        except Exception:
            pass

    return best_x
