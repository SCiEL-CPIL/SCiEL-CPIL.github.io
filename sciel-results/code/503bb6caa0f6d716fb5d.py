import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution
from math import lgamma, log, pi

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl) # shape (70,)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()
Y_flat = Y_basis.reshape(21, -1)

def objective_and_grad(coeffs):
    """
    Computes the objective function R = ||f||_4 / ||f||_2 and its gradient.
    coeffs: shape (21,)
    Returns: (value, gradient)
    """
    f_flat = coeffs @ Y_flat
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return 1e10, np.zeros(21)
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    val = norm4 / norm2
    
    term1_coeff = (norm4_4 ** -0.75) * (norm2_sq ** -0.5)
    term2_coeff = (norm4_4 ** 0.25) * (norm2_sq ** -1.5)
    
    grad1 = Y_flat.T @ (W_flat * (Y_flat * f_flat**2))
    grad2 = Y_flat.T @ (W_flat * (Y_flat * f_flat))
    
    grad = term1_coeff * grad1 - term2_coeff * grad2
    
    return val, grad

def objective_only(coeffs):
    """
    Computes only the objective function R = ||f||_4 / ||f||_2.
    coeffs: shape (21,)
    Returns: value
    """
    val, _ = objective_and_grad(coeffs)
    return val

def run_search(time_budget):
    start_time = time.time()
    best_coeffs = None
    best_val = np.inf
    
    # Helper to check time
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # 1. Initial global search using Differential Evolution
    # Bounds for coefficients: we can scale them arbitrarily, so we bound the norm.
    # The ratio is scale-invariant, but DE works better with bounded parameters.
    # We'll search in a box and then normalize or just let it find the direction.
    # Actually, since the ratio is homogeneous of degree 0, we can fix one coefficient 
    # or normalize, but DE doesn't handle constraints well. 
    # Instead, we'll search in a reasonable range and rely on the fact that 
    # the objective is invariant under scaling. We can just bound coefficients 
    # to [-10, 10] for example.
    
    bounds = [(-10.0, 10.0) for _ in range(21)]
    
    # Run DE with a time limit
    if time_left() > 100.0:
        try:
            de_result = differential_evolution(
                objective_only,
                bounds,
                maxiter=100,
                tol=1e-6,
                popsize=20,
                seed=42,
                polish=False, # We will polish later
                callback=lambda xk, convergence: (time.time() - start_time) < (time_budget - 50.0)
            )
            
            if de_result.success or de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception as e:
            pass
    
    # 2. Local refinement using L-BFGS-B from the best DE point
    if best_coeffs is not None and time_left() > 30.0:
        try:
            result = minimize(
                objective_and_grad, 
                best_coeffs, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success:
                val_res, grad_res = objective_and_grad(result.x)
                if val_res < best_val:
                    best_val = val_res
                    best_coeffs = result.x.copy()
        except Exception:
            pass
            
    # 3. If we still have time, try a few random restarts with local optimization
    if time_left() > 60.0:
        for i in range(5):
            if time_left() < 20.0:
                break
            c0 = np.random.randn(21)
            try:
                result = minimize(
                    objective_and_grad, 
                    c0, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
                
                if result.success:
                    val_res, _ = objective_and_grad(result.x)
                    if val_res < best_val:
                        best_val = val_res
                        best_coeffs = result.x.copy()
            except Exception:
                continue
            
    return best_coeffs

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Time budget management
    start_time = time.time()
    max_time = 580.0 # Safety margin from 630s (leave 50s buffer)
    
    # Run search
    try:
        best_coeffs = run_search(max_time)
    except Exception as e:
        # Fallback to baseline if something goes wrong
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    # Ensure size is 21
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Check max abs coefficient constraints
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Normalize to be within range, e.g., scale to max 1.0
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs
