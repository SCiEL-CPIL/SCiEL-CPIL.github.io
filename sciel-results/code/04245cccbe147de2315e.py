"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as the square of an even Chebyshev polynomial series.
f(x) = (sum c_k T_{2k}(4x))^2.
This enforces non-negativity and symmetry.
Optimize the coefficients c_k using L-BFGS-B.
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

def get_even_chebyshev_basis(x, max_order):
    """
    Generate basis for even Chebyshev polynomials T_{2k}(4x).
    x is in [-1/4, 1/4]. 4x is in [-1, 1].
    """
    u = 4.0 * x  # Map [-0.25, 0.25] to [-1, 1]
    u = np.clip(u, -1.0, 1.0)
    
    n_terms = max_order + 1
    basis = np.zeros((len(x), n_terms))
    
    # T_0(u) = 1
    basis[:, 0] = 1.0
    
    if n_terms > 1:
        # We need T_2, T_4, ..., T_{2*max_order}
        # T_0(u) = 1
        # T_1(u) = u
        # T_k(u) = 2u T_{k-1}(u) - T_{k-2}(u)
        
        T_k_minus_2 = np.ones_like(u) # T_0
        T_k_minus_1 = u               # T_1
        
        # Compute T_2
        T_2 = 2 * u * T_k_minus_1 - T_k_minus_2
        basis[:, 1] = T_2
        
        for k in range(3, 2 * max_order + 1):
            T_k = 2 * u * T_k_minus_1 - T_k_minus_2
            if k % 2 == 0:
                idx = k // 2
                if idx < n_terms:
                    basis[:, idx] = T_k
            T_k_minus_2 = T_k_minus_1
            T_k_minus_1 = T_k
            
    return basis

def get_f_from_coeffs(coeffs, basis):
    """Compute f(x) = (sum c_k T_{2k}(u))^2"""
    p = basis @ coeffs
    return p * p

def objective(coeffs, basis):
    """
    Objective function: normalized max autoconvolution.
    """
    f = get_f_from_coeffs(coeffs, basis)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = np.convolve(f, f, mode='full')
    score = np.max(conv) / (s * s)
    return score

def construct():
    start_time = time.time()
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Initial guess: Gaussian-like function
    # f(x) = exp(-alpha * x^2)
    # We want P(x) = sqrt(f(x)) = exp(-0.5 * alpha * x^2)
    # We will fit this to the Chebyshev basis or just use it as a starting point for coefficients.
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders for the even polynomial P(x)
    # f(x) = P(x)^2. P(x) is even.
    for max_order in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        n_params = max_order + 1
        basis = get_even_chebyshev_basis(x_grid, max_order)
        
        best_params = None
        best_obj = np.inf
        
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            # Initialization
            if restart == 0:
                # Approximate a Gaussian P(x) = exp(-2 * x^2) using least squares on the basis
                # x is in [-0.25, 0.25]
                x_test = np.linspace(-0.25, 0.25, 200)
                basis_test = get_even_chebyshev_basis(x_test, max_order)
                target = np.exp(-2 * x_test**2)
                # Solve least squares
                coeffs_init, residuals, rank, s = np.linalg.lstsq(basis_test, target, rcond=None)
                x0 = coeffs_init
            else:
                # Random small perturbation around the best found so far or constant
                if best_params is not None:
                    rng = np.random.RandomState(42 + restart)
                    x0 = best_params + rng.normal(0, 0.1, n_params)
                else:
                    x0 = np.zeros(n_params)
                    x0[0] = 1.0
                    if n_params > 1:
                        x0[1] = -0.5 # Slight dip
            
            # Optimize using L-BFGS-B
            try:
                result = minimize(
                    objective, x0, args=(basis,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                f_test = get_f_from_coeffs(result.x, basis)
                actual_score = compute_score(f_test)
                
                if actual_score < best_obj:
                    best_obj = actual_score
                    best_params = result.x.copy()
                    
            except Exception:
                pass
        
        if best_params is not None:
            f = get_f_from_coeffs(best_params, basis)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score

    # Local refinement on the grid
    if time.time() - start_time < TIME_LIMIT - 20:
        current_h = best_f.copy()
        
        # Normalize to sum = N for numerical stability in local search
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        # Simple local search: perturb random indices
        for outer_iter in range(100):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            # Pick a few random indices to perturb
            for _ in range(30):
                j = np.random.randint(0, N)
                old_val = current_h[j]
                
                # Try scaling
                for scale in [0.9, 0.8, 1.1, 1.2, 0.7, 1.3]:
                    h_new = current_h.copy()
                    h_new[j] = old_val * scale
                    h_new = np.maximum(h_new, 0.0)
                    
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
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
