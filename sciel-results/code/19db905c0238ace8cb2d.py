"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h (circular shift)."""
    n = len(h)
    # h is a numpy array
    # Circular correlation
    # R_k = sum h[i] h[(i+k)%n] * (2/n)
    # C_k = 1 - R_k
    # We want max C_k = 1 - min R_k
    min_corr = float('inf')
    for k in range(1, n):
        # Circular shift
        h_shifted = np.roll(h, k)
        corr = np.dot(h, h_shifted) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return 1.0 - min_corr

def _objective_soft(x, beta=100.0):
    """Soft min of correlations."""
    n = len(x)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.roll(x, k)
        corrs[k-1] = np.dot(x, h_shifted) * (2.0 / n)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return 1.0 - soft_min

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient for soft min."""
    n = len(x)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.roll(x, k)
        corrs[k-1] = np.dot(x, h_shifted) * (2.0 / n)
        
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w != 0:
            h_shifted = np.roll(x, k)
            # d/dx [ sum x[i] x[i+k] ] = x[i+k] + x[i-k] (circular)
            # Actually, d/dx_j sum_i x_i x_{i+k} = x_{j+k} + x_{j-k}
            grad += w * (np.roll(x, -k) + np.roll(x, k))
            
    grad_obj = -grad
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    for _ in range(10):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0: break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0: break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    return w

def construct():
    n = N
    target_sum = n / 2.0
    
    # Start with the theoretically optimal constant solution
    best_x = np.full(n, 0.5)
    best_obj = _compute_max_overlap(best_x) # Should be 0.75
    
    # Try to improve with optimization
    # Since constant 0.5 is likely optimal, we do a quick check
    # If we find something better, we use it.
    
    # Use a few random restarts to be sure
    np.random.seed(42)
    restarts = [best_x]
    for _ in range(5):
        restarts.append(np.random.uniform(0, 1, n))
        
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        x = _project_to_simplex(x0, target_sum)
        try:
            res = minimize(
                _objective_and_grad_soft,
                x,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = _compute_max_overlap(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
            
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
            
    return best_x
# EVOLVE-BLOCK-END
