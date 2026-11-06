"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _pairwise_distances(C):
    """Compute pairwise distances between all centers."""
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _soft_min_radius(C, beta=50.0):
    """
    Smooth surrogate for the packing radius using log-sum-exp.
    r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|))
    We maximize r, so we minimize -r.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise distances / 2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Boundary distances: 1/2 - |c_ik| for each coordinate
    # For each center i and coord k: 0.5 - |C[i,k]|
    abs_C = np.abs(C)
    boundary_terms = 0.5 - abs_C  # shape (n, 3)
    
    # Combine all constraints into one vector
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], 
                                 boundary_terms.flatten()])
    
    # Soft min: (1/beta) * log(sum(exp(-beta * x))) approximates min(x) for large beta
    # We want to maximize min, so minimize -soft_min
    # soft_min ≈ (1/beta) * log(sum(exp(-beta * x)))
    # -soft_min ≈ -(1/beta) * log(sum(exp(-beta * x)))
    
    # Numerically stable: subtract max before exp
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    return -soft_min_val  # We minimize this (maximize radius)


def _grad_soft_min_radius(C, beta=50.0):
    """Gradient of the soft-min radius surrogate."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, 1.0)
    pair_terms = dists / 2.0
    
    abs_C = np.abs(C)
    boundary_terms = 0.5 - abs_C
    
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], 
                                 boundary_terms.flatten()])
    
    # Soft min weights
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of soft_min w.r.t. all_terms is -weights
    # d(soft_min)/d(all_terms) = -weights
    # We're minimizing -soft_min, so gradient of objective = weights
    
    grad_all = np.zeros_like(all_terms)
    grad_all[:] = weights
    
    # Now backprop to C
    grad_C = np.zeros_like(C)
    
    # Pairwise part
    n_pairs = n * (n - 1) // 2
    pair_grad = grad_all[:n_pairs]
    triu_idx = np.triu_indices(n, k=1)
    
    # For each pair (i,j), term = |C_i - C_j| / 2
    # d(term)/d(C_i) = (C_i - C_j) / (2 * |C_i - C_j|) = (C_i - C_j) / (4 * pair_term)
    # d(term)/d(C_j) = (C_j - C_i) / (2 * |C_i - C_j|) = -(C_i - C_j) / (4 * pair_term)
    
    for idx in range(n_pairs):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        d = C[i] - C[j]
        dist = np.sqrt(np.sum(d ** 2) + 1e-30)
        if dist < 1e-15:
            continue
        coeff = pair_grad[idx] / (2.0 * dist)
        grad_C[i] += coeff * d
        grad_C[j] -= coeff * d
    
    # Boundary part
    bnd_grad = grad_all[n_pairs:]
    bnd_grad_reshaped = bnd_grad.reshape(n, 3)
    # term = 0.5 - |C[i,k]|, d(term)/d(C[i,k]) = -sign(C[i,k])
    signs = np.sign(C)
    signs[abs_C < 1e-15] = 1.0
    grad_C += -bnd_grad_reshaped * signs
    
    return grad_C


def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    abs_C = np.abs(C)
    min_bnd = np.min(0.5 - abs_C)
    
    return min(min_pair, min_bnd)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    C0 = grid[:N].copy()
    
    best_C = C0.copy()
    best_r = _exact_radius(C0)
    
    # Try multiple betas (from smooth to sharp)
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
        
        def objective(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta)
        
        def gradient(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta)
            return g.flatten()
        
        x0 = best_C.flatten()
        
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Final high-precision polish with a very sharp beta
    if time.time() < deadline - 10.0:
        beta_final = 500.0
        
        def objective_final(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta_final)
        
        def gradient_final(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta_final)
            return g.flatten()
        
        try:
            res = minimize(
                objective_final,
                best_C.flatten(),
                jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube and check
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
# EVOLVE-BLOCK-END
