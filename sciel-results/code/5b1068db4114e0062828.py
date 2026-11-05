"""Rewritten program for QAOA parameter design with optimized evaluation and Nelder-Mead search."""
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

def _apply_mixer_fast(psi, n, beta):
    """
    Apply exp(-i * beta * sum(X_i)) to the state vector psi in-place.
    Uses a faster approach with reshape and broadcasting.
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    shape = (2,) * n
    psi_t = psi.reshape(shape)
    
    for i in range(n):
        # Move axis i to front for easier slicing
        psi_moved = np.moveaxis(psi_t, i, 0)
        half_0 = psi_moved[0].copy()
        half_1 = psi_moved[1].copy()
        psi_moved[0] = cb * half_0 + sb * half_1
        psi_moved[1] = sb * half_0 + cb * half_1
        # Move axis back
        psi_t = np.moveaxis(psi_moved, 0, i)
    
    return psi_t.reshape(-1)

def _qaoa_expectation(n, edges, gammas, betas, cut_vals):
    """
    Compute the expected cut size.
    gammas, betas are arrays of length p.
    """
    dim = 1 << n
    # Initial state |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    
    for k in range(p):
        # 1. Apply Cost Unitary: exp(-i * gamma_k * C)
        # C is diagonal with entries cut_vals
        # Multiplier is exp(-1j * gamma_k * cut_vals)
        multipliers = np.exp(-1j * gammas[k] * cut_vals)
        psi *= multipliers
        
        # 2. Apply Mixer Unitary: exp(-i * beta_k * sum(X_i))
        psi = _apply_mixer_fast(psi, n, betas[k])
        
    # Calculate expectation
    # <C> = sum |psi_i|^2 * cut_vals_i
    probs = np.abs(psi) ** 2
    exp_cut = np.dot(probs, cut_vals)
    
    return exp_cut

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using Nelder-Mead with multiple restarts and a final polish.
    """
    # Precompute cut values
    cut_vals = _precompute_cut_values(n, edges)
    
    # Objective function for minimization (negative expectation)
    def objective(params):
        # params: [g0, b0, g1, b1]
        g0, b0, g1, b1 = params
        gammas = np.array([g0, g1])
        betas = np.array([b0, b1])
        try:
            exp_cut = _qaoa_expectation(n, edges, gammas, betas, cut_vals)
            return -exp_cut
        except Exception:
            return 1e9 # Large penalty for errors

    best_score = -np.inf
    best_params = None
    
    # Number of restarts
    # Since evaluation is fast, we can do many.
    # Let's estimate: 1 eval ~ 50us. 1 Nelder-Mead run ~ 100 evals = 5ms.
    # In 100s, we can do ~20,000 runs? No, overhead. Let's do 100-500 runs.
    num_restarts = 200
    
    np.random.seed(42)
    
    for r in range(num_restarts):
        # Check time
        if time.time() - start_time > timeout - 10:
            break
            
        # Random initialization
        # gamma in [0, pi], beta in [0, pi]
        init = np.random.uniform(0, np.pi, size=4)
        
        # Nelder-Mead options
        options = {
            'maxiter': 100,
            'xatol': 1e-4,
            'fatol': 1e-4
        }
        
        try:
            res = scipy.optimize.minimize(objective, init, method='Nelder-Mead', options=options)
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
            
    # Final high-precision polish pass on the best parameters found
    if best_params is not None and time.time() - start_time < timeout - 2:
        try:
            # Tighter tolerance, more iterations
            polish_options = {
                'maxiter': 500,
                'xatol': 1e-7,
                'fatol': 1e-7
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
        # Reshape to (p, 2)
        # best_params is [g0, b0, g1, b1]
        # We need [[g0, b0], [g1, b1]]
        # Note: scipy returns flat array.
        # My objective takes [g0, b0, g1, b1]
        # So best_params[0]=g0, best_params[1]=b0, best_params[2]=g1, best_params[3]=b1
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        # Fallback
        return np.array([[0.35, 0.25], [0.35, 0.25]])

def angles(n, edges, p):
    start_time = time.time()
    timeout = 145 # Safety margin from 150s limit
    
    # Optimize angles
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    # Final check
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
