import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. Use L-BFGS-B on the scale-invariant ratio J(c) = ||Yc||_4^4 / ||Yc||_2^4.
    3. Multiple restarts from diverse initializations.
    4. Final refinement pass with tighter tolerances.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590
    
    # --- 1. Precompute Spherical Harmonics ---
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Normalization constants
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # Precompute Y^T W for fast gradients
    YT_W = Y_matrix.T @ W_eff  # (21, 9100) -- actually this is wrong shape
    # Let's just do Y^T @ (W * vec) on the fly. It's fast enough.
    
    # --- 2. Define Objective and Gradient ---
    def objective_and_grad(c):
        # c is shape (21,)
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)
        
        if D_val < 1e-12:
            return 1e10, np.zeros_like(c)
            
        N_val = np.dot(W_eff, f2 * f2)
        
        val = N_val / (D_val * D_val)
        
        # Gradient
        # grad_N = 4 * Y^T (W * f^3)
        # grad_D = 2 * Y^T (W * f)
        
        f3 = f2 * f
        grad_N = 4.0 * (Y_matrix.T @ (W_eff * f3))
        grad_D = 2.0 * (Y_matrix.T @ (W_eff * f))
        
        # val = N / D^2
        # dval/dc = (dN D^2 - N 2 D dD) / D^4 = (dN D - 2 N dD) / D^3
        
        grad = (grad_N * D_val - 2.0 * N_val * grad_D) / (D_val ** 3)
        
        return val, grad

    # --- 3. Optimization ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    inits = []
    
    # 1. Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # 2. Symmetric pairs
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 1.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = -1.0
        inits.append(init)
        
    # 3. Random vectors
    for _ in range(100):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # 4. Sparse random
    for _ in range(100):
        init = np.random.randn(n_coeffs)
        mask = np.random.rand(n_coeffs) < 0.3
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    # 5. Anti-symmetric patterns
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 0.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 0.0
        init[-m + k] = 1.0
        inits.append(init)
        
    np.random.shuffle(inits)
    
    lbfgs_options = {
        'maxiter': 2000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 20,
    }
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', options=lbfgs_options)
            
            # Normalize result to unit norm for consistency
            c_res = res.x
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            # Evaluate final score
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
                
        except Exception:
            continue

    # --- 4. Refine Best Solution ---
    if time.time() < deadline - 10:
        try:
            lbfgs_options_refine = {
                'maxiter': 10000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 50,
            }
            res = minimize(objective_and_grad, best_coeffs, jac=True, method='L-BFGS-B', options=lbfgs_options_refine)
            
            c_res = res.x
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
        except Exception:
            pass

    # --- 5. Validate and Return ---
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
