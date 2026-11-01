import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    """Generate hexagonal lattice initialization for circle packing."""
    # Spacing s such that we get at least n points in unit circle
    # Approx area per point s^2 * sqrt(3)/2. Total area pi.
    # n * s^2 * sqrt(3)/2 <= pi -> s <= sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95  # 5% margin
    
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
        # Fallback: scale down to fit
        r_max = np.max(points[:, 2])
        scale = 1.0 / r_max
        return points[:, :2] * scale

def _calc_score(x, idx_i, idx_j):
    """Calculate the true packing score: min(min pairwise distance/2, min boundary distance)."""
    diff = x[idx_i] - x[idx_j]
    d_ij = np.linalg.norm(diff, axis=1)
    min_d_ij = np.min(d_ij) / 2.0
    
    r_i = np.linalg.norm(x, axis=1)
    min_d_bound = np.min(1.0 - r_i)
    
    return np.min([min_d_ij, min_d_bound])

def _make_objective(idx_i, idx_j, beta):
    """Create objective function and gradient for soft-min optimization with given beta."""
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
        
        r_i = np.linalg.norm(x, axis=1)
        d_bound = 1.0 - r_i
        
        y_pairs = d_ij / 2.0
        y_bound = d_bound
        all_y = np.concatenate([y_pairs, y_bound])
        y_min = np.min(all_y)
        
        exp_terms = np.exp(-beta * (all_y - y_min))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
        
        grad = np.zeros_like(x)
        w_pairs = weights[:len(y_pairs)]
        terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
        np.add.at(grad, idx_i, -terms)
        np.add.at(grad, idx_j, terms)
        
        w_bound = weights[len(y_pairs):]
        r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
        grad_bound = -x / r_i_safe[:, np.newaxis]
        grad -= (w_bound[:, np.newaxis] * grad_bound)
        
        return obj_val, grad
    
    return objective_and_grad

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    x0 = _hex_lattice_init(N)
    idx_i, idx_j = np.triu_indices(N, 1)
    
    best_x = x0.copy()
    best_score = _calc_score(best_x, idx_i, idx_j)
    
    # Multi-stage optimization with increasing beta
    beta_schedule = [50, 100, 200, 500, 1000]
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
        
        obj_func = _make_objective(idx_i, idx_j, beta)
        
        try:
            res = minimize(
                obj_func, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 80, 'gtol': 1e-9}
            )
            
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                # Validate: check for non-finite values
                if np.all(np.isfinite(candidate_x)):
                    score = _calc_score(candidate_x, idx_i, idx_j)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    # Final polish with very high beta if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            obj_func_polish = _make_objective(idx_i, idx_j, 5000)
            res = minimize(
                obj_func_polish, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 30, 'gtol': 1e-11}
            )
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    score = _calc_score(candidate_x, idx_i, idx_j)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    return best_x
