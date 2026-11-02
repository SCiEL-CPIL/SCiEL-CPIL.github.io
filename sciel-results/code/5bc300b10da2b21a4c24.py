"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 150s kill
    
    # Baseline: Flat function
    best_f = np.ones(N)
    
    def compute_value(f):
        """Compute normalized max autoconvolution."""
        f = np.asarray(f, dtype=np.float64)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        # Normalized maximum: max(f*f) / (sum(f))^2 * N
        return max_conv / (s ** 2) * N

    best_val = compute_value(best_f)
    
    # Use DCT basis representation to reduce dimensionality
    # f = sum_k c_k * basis_k, where basis_k are DCT modes
    # We use a small number of DCT modes (e.g., 30)
    n_modes = 30
    
    # Precompute DCT basis vectors
    # DCT-II basis: basis_k[i] = cos(pi * k * (2*i + 1) / (2*N))
    # Normalize so that basis vectors are orthonormal
    i = np.arange(N)
    k = np.arange(n_modes)
    # Shape: (n_modes, N)
    basis = np.cos(np.pi * k[:, None] * (2 * i[None, :] + 1) / (2 * N))
    # Normalize each basis vector
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    basis = basis / norms[:, None]
    
    def coeffs_to_f(c):
        """Convert non-negative coefficients to function values."""
        # c has shape (n_modes,)
        # f = c @ basis
        f = c @ basis
        # Ensure non-negative
        f = np.maximum(f, 0)
        return f
    
    def compute_value_from_coeffs(c):
        """Compute value from coefficients."""
        f = coeffs_to_f(c)
        return compute_value(f)
    
    # Objective function with smooth max approximation
    def objective(c):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return best_val
        f = coeffs_to_f(c)
        s = np.sum(f)
        if s < 1e-12:
            return best_val
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        return max_conv / (s ** 2) * N
    
    # Gradient of objective w.r.t coefficients
    def objective_grad(c):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return np.zeros_like(c)
        f = coeffs_to_f(c)
        s = np.sum(f)
        if s < 1e-12:
            return np.zeros_like(c)
        
        # Use smooth max approximation for differentiability
        beta = 30.0
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        log_sum_exp = np.log(np.sum(exp_conv)) + beta * max_conv
        obj = log_sum_exp / beta
        
        # Gradient of obj w.r.t conv
        weights = exp_conv / np.sum(exp_conv)
        d_obj_d_conv = weights / beta
        
        # Gradient w.r.t f using FFT
        L = 4 * N
        F = np.fft.fft(f, L)
        U = np.fft.fft(d_obj_d_conv, L)
        d_obj_d_f_full = np.fft.ifft(U * (np.conj(F) + F)).real
        d_obj_d_f = d_obj_d_f_full[:N]
        
        # Gradient w.r.t c: d_obj/d_c_k = sum_i (d_obj/d_f_i) * basis_k[i]
        # But f = c @ basis, so d_f_i/d_c_k = basis[k, i]
        grad_c = basis @ d_obj_d_f
        
        return grad_c
    
    # Optimize using L-BFGS-B with non-negativity constraints
    # Initial guess: small positive values
    c0 = np.ones(n_modes) * 0.1
    
    constraints = []
    bounds = [(0, None)] * n_modes
    
    try:
        result = minimize(
            objective, c0,
            jac=objective_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success:
            f_opt = coeffs_to_f(result.x)
            val = compute_value(f_opt)
            if val < best_val:
                best_val = val
                best_f = f_opt.copy()
    except Exception:
        pass
    
    # Try multiple random restarts
    n_restarts = 5
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        c0 = np.abs(np.random.randn(n_modes)) * 0.1 + 0.01
        
        try:
            result = minimize(
                objective, c0,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success:
                f_opt = coeffs_to_f(result.x)
                val = compute_value(f_opt)
                if val < best_val:
                    best_val = val
                    best_f = f_opt.copy()
        except Exception:
            continue
    
    result_heights = np.maximum(best_f, 0)
    return [float(h) for h in result_heights]
# EVOLVE-BLOCK-END
