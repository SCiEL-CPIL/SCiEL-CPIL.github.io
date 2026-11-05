"""Combined QAOA parameter design: vectorized simulation + analytic gradients + L-BFGS-B with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _precompute_cut_values(n, edges):
    """
    Precompute the cut value for every possible state mask.
    Returns an array of shape (2^n,).
    """
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals

def _precompute_complement_indices(n):
    """
    Precompute the bitwise complement indices for mixer application.
    """
    dim = 1 << n
    mask = dim - 1
    return np.array([i ^ mask for i in range(dim)], dtype=np.int64)

def _apply_mixer_fast(psi, beta, comp_indices):
    """
    Apply exp(-i * beta * sum(X_i)) to the state vector psi.
    Uses the property: exp(-i beta X)|x> = cos(beta)|x> - i sin(beta)|x^>
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    return cb * psi + sb * psi[comp_indices]

def _qaoa_state(n, gammas, betas, cut_vals, comp_indices):
    """
    Compute the final state vector after QAOA evolution.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    for k in range(p):
        psi *= np.exp(-1j * gammas[k] * cut_vals)
        psi = _apply_mixer_fast(psi, betas[k], comp_indices)
        
    return psi

def _qaoa_expectation(n, gammas, betas, cut_vals, comp_indices):
    """
    Compute the expected cut size.
    """
    psi = _qaoa_state(n, gammas, betas, cut_vals, comp_indices)
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_vals))

def _qaoa_gradient_param_shift(n, params, cut_vals, comp_indices):
    """
    Compute gradient using parameter shift rule.
    params = [g0, b0, g1, b1]
    """
    grad = np.zeros(4)
    shift = np.pi / 2.0
    
    # For each parameter, compute f(x+shift) and f(x-shift)
    for i in range(4):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift
        
        g_plus = np.array([params_plus[0], params_plus[2]])
        b_plus = np.array([params_plus[1], params_plus[3]])
        g_minus = np.array([params_minus[0], params_minus[2]])
        b_minus = np.array([params_minus[1], params_minus[3]])
        
        f_plus = _qaoa_expectation(n, g_plus, b_plus, cut_vals, comp_indices)
        f_minus = _qaoa_expectation(n, g_minus, b_minus, cut_vals, comp_indices)
        
        grad[i] = 0.5 * (f_plus - f_minus)
        
    return grad

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using L-BFGS-B with gradient, followed by Nelder-Mead polish.
    """
    cut_vals = _precompute_cut_values(n, edges)
    comp_indices = _precompute_complement_indices(n)
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation(n, gammas, betas, cut_vals, comp_indices)
        except Exception:
            return 1e9

    def gradient(params):
        try:
            return -_qaoa_gradient_param_shift(n, params, cut_vals, comp_indices)
        except Exception:
            return np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    # Estimate evaluation time
    t0 = time.time()
    _qaoa_expectation(n, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals, comp_indices)
    t1 = time.time()
    eval_time = max(t1 - t0, 0.0001)
    
    # L-BFGS-B with gradient: ~100 iterations, each ~10 evals (grad + func)
    # Gradient takes 4 evals, func takes 1. Total 5 per step.
    time_per_restart_lbfgs = 100 * 5 * eval_time
    available_time = timeout - (time.time() - start_time) - 5
    num_restarts_lbfgs = max(5, min(50, int(available_time / max(time_per_restart_lbfgs, 0.001))))
    
    np.random.seed(42)
    
    base_inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.6, 0.4, 0.6],
        [0.6, 0.2, 0.6, 0.2],
    ]
    
    for r in range(num_restarts_lbfgs):
        if time.time() - start_time > timeout - 10:
            break
            
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform([0.1, 0.1, 0.1, 0.1], [1.5, 1.5, 1.5, 1.5])
        
        try:
            res = scipy.optimize.minimize(
                objective, init, jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.0, np.pi)] * 4,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
            
    # Polish with Nelder-Mead from the best L-BFGS-B result
    if best_params is not None and time.time() - start_time < timeout - 5:
        try:
            polish_options = {
                'maxiter': 200,
                'xatol': 1e-8,
                'fatol': 1e-8
            }
            res_polish = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead', options=polish_options
            )
            polish_score = -res_polish.fun
            if polish_score > best_score:
                best_score = polish_score
                best_params = res_polish.x
        except Exception:
            pass

    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])

def angles(n, edges, p):
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
