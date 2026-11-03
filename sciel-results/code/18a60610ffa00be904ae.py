"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as a sum of even-order Chebyshev polynomials
with nonnegative coefficients to ensure non-negativity. Optimize using a
minimax approach where we directly minimize the peak of the autoconvolution.
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
    x should be in [-1/4, 1/4]. Returns array of shape (len(x), n_terms)
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

def params_to_step_nonneg(params, max_order):
    """
    Convert nonnegative polynomial coefficients to a step function.
    params: array of nonnegative coefficients for even Chebyshev orders 0, 2, 4, ...
    Returns nonnegative step function of length N.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    basis = chebyshev_even_basis(x, max_order)
    
    # Ensure params are nonnegative
    params = np.maximum(params, 0.0)
    
    f = basis @ params
    f = np.maximum(f, 0.0)
    return f

def minimax_objective(params, max_order, x_grid, basis):
    """
    Minimax objective: minimize the maximum of the autoconvolution.
    Uses the fact that we can compute the autoconvolution in the frequency domain
    for efficiency.
    """
    f = params_to_step_nonneg(params, max_order)
    
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    # Compute autoconvolution
    conv = np.convolve(f, f, mode='full')
    
    val = np.max(conv) / (s * s)
    return val

def construct():
    start_time = time.time()
    
    # Precompute grid and basis for common max_orders
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Start with constant function (all ones)
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders for Chebyshev basis
    # Even orders: 0, 2, 4, 6, 8, 10, 12, 14, 16
    for max_order in [2, 4, 6, 8, 10, 12, 14, 16]:
        if time.time() - start_time > TIME_LIMIT - 25:
            break
        
        n_params = max_order // 2 + 1
        basis = chebyshev_even_basis(x_grid, max_order)
        
        best_params = None
        best_obj = np.inf
        
        # Multiple restarts with different initializations
        n_restarts = 12
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            rng = np.random.RandomState(42 + max_order * 100 + restart)
            
            # Different initialization strategies
            strategy = restart % 4
            
            if strategy == 0:
                # Dominant constant term with small perturbations
                x0 = np.zeros(n_params)
                x0[0] = 1.0
                for i in range(1, n_params):
                    x0[i] = rng.uniform(0, 0.2)
            elif strategy == 1:
                # Uniform small coefficients
                x0 = rng.uniform(0.05, 0.5, n_params)
            elif strategy == 2:
                # Decay with order
                x0 = np.array([1.0 / (1 + i) for i in range(n_params)])
                x0 = x0 + rng.uniform(0, 0.1, n_params)
            else:
                # Random but ensuring constant term is dominant
                x0 = rng.uniform(0, 0.3, n_params)
                x0[0] = 1.0
        
            # Optimize using L-BFGS-B with bounds [0, inf)
            try:
                result = minimize(
                    minimax_objective, x0, args=(max_order, x_grid, basis),
                    method='L-BFGS-B',
                    bounds=[(0, None)] * n_params,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_obj:
                    f_test = params_to_step_nonneg(result.x, max_order)
                    actual_score = compute_score(f_test)
                    if actual_score < best_obj:
                        best_obj = actual_score
                        best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is not None:
            f = params_to_step_nonneg(best_params, max_order)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: coordinate descent on the full 600-dim space
    if time.time() - start_time < TIME_LIMIT - 20:
        current_h = best_f.copy()
        
        # Normalize to have sum = N
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        # Coordinate descent: try adjusting each coordinate
        for outer_iter in range(30):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            for j in range(N):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                # Try a few values for coordinate j
                old_val = current_h[j]
                
                # Try reducing the peak by adjusting this coordinate
                # The peak of autoconvolution is at some lag; reducing f[j] 
                # reduces contributions to conv at lags involving j
                
                # Simple heuristic: try scaling down high values
                if old_val > np.mean(current_h) * 1.2:
                    for scale in [0.8, 0.7, 0.6, 0.5]:
                        h_new = current_h.copy()
                        h_new[j] = old_val * scale
                        
                        s_new = np.sum(h_new)
                        if s_new < 1e-12:
                            continue
                        h_new = h_new * (N / s_new)
                        
                        conv_new = np.convolve(h_new, h_new, mode='full')
                        val_new = np.max(conv_new) / (N * N)
                        
                        if val_new < best_score:
                            current_h = h_new
                            best_score = val_new
                            best_f = current_h.copy()
                            improved = True
                            break
            if not improved:
                break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
