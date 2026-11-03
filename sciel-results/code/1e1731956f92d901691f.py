"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as a low-rank sum of Kronecker products
of 1D Chebyshev polynomials, exploiting symmetry and smoothness. Optimize the
small number of polynomial coefficients using a trust-region method with a
log-barrier for non-negativity.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def chebyshev_even_basis(x, max_order):
    """
    Generate Chebyshev polynomial basis values for even orders only.
    x should be in [-1, 1]. Returns array of shape (len(x), n_terms)
    where n_terms = max_order//2 + 1 (orders 0, 2, 4, ..., max_order if even)
    """
    # Map x from [-1/4, 1/4] to [-1, 1] for Chebyshev
    x_scaled = x / 0.25  # now in [-1, 1]
    
    n_terms = max_order // 2 + 1
    basis = np.zeros((len(x), n_terms))
    
    for i in range(n_terms):
        order = 2 * i
        basis[:, i] = np.cos(order * np.arccos(np.clip(x_scaled, -1, 1)))
    
    return basis

def params_to_step(params, max_order):
    """
    Convert polynomial coefficients to a step function.
    params: array of coefficients for even Chebyshev orders 0, 2, 4, ...
    Returns nonnegative step function of length N.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    basis = chebyshev_even_basis(x, max_order)
    
    f = basis @ params
    f = np.maximum(f, 0.0)
    return f

def barrier_objective(params, max_order, mu=0.1):
    """
    Objective with log-barrier for non-negativity.
    We minimize the normalized max autoconvolution plus a barrier term
    that penalizes values close to zero.
    """
    f = params_to_step(params, max_order)
    
    # Barrier term: -mu * sum(log(f + eps)) to keep f away from 0
    eps = 1e-10
    barrier = -mu * np.sum(np.log(f + eps))
    
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return 1e10 + barrier
    
    val = np.max(conv) / (s * s)
    return val + barrier

def construct():
    start_time = time.time()
    
    # Start with constant function (all ones)
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders for Chebyshev basis
    # Even orders: 0, 2, 4, 6, 8, 10, 12
    for max_order in [2, 4, 6, 8, 10, 12]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        n_params = max_order // 2 + 1
        
        # Initial guess: constant + small perturbations
        rng = np.random.RandomState(42 + max_order)
        
        best_params = None
        best_obj = np.inf
        
        # Multiple restarts with different initializations
        for restart in range(8):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Initialize: start with dominant constant term, small others
            x0 = np.zeros(n_params)
            x0[0] = 1.0  # constant term
            for i in range(1, n_params):
                x0[i] = rng.uniform(-0.3, 0.3)
            
            # Try with barrier method
            for mu in [0.05, 0.1, 0.2]:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                try:
                    result = minimize(
                        barrier_objective, x0, args=(max_order, mu),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    
                    if result.success and result.fun < best_obj:
                        # Check if the solution is valid (nonnegative)
                        f_test = params_to_step(result.x, max_order)
                        if np.all(f_test >= -1e-10):
                            actual_score = compute_score(f_test)
                            if actual_score < best_obj or best_params is None:
                                best_obj = actual_score
                                best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = params_to_step(best_params, max_order)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: gradient-based optimization on the full 600-dim space
    # starting from the best parametric solution
    if time.time() - start_time < TIME_LIMIT - 15:
        # Use the parametric solution as starting point for local refinement
        current_h = best_f.copy()
        
        # Simple projected gradient descent to reduce peak
        alpha = 0.02
        for iteration in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            s = np.sum(current_h)
            if s < 1e-12:
                break
            
            # Normalize
            current_h = current_h * (N / s)
            
            conv = np.convolve(current_h, current_h, mode='full')
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Compute gradient of max(conv) w.r.t. current_h
            # Use subgradient: at the maximizing lag, gradient is current_h flipped
            peak_lag = np.argmax(conv)
            
            # d(conv[peak_lag])/d(current_h[j]) = current_h[peak_lag - j]
            grad = np.zeros(N)
            j_min = max(0, peak_lag - (N - 1))
            j_max = min(N - 1, peak_lag)
            for j in range(j_min, j_max + 1):
                grad[j] = current_h[peak_lag - j]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Projected gradient step: reduce where gradient is high
            h_new = current_h - alpha * grad
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            conv_new = np.convolve(h_new, h_new, mode='full')
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                current_h = h_new
                if val_new < best_score:
                    best_score = val_new
                    best_f = current_h.copy()
                alpha = min(alpha * 1.05, 0.2)
            else:
                alpha *= 0.8
                if alpha < 1e-6:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
