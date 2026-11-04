import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

def _compute_ratio_lse(h, n_grid=4096, tau=1e-3):
    """
    Compute the LSE-approximated ratio: max(autoconv) / (int f)^2.
    h: array of step heights (length N)
    n_grid: resolution for fine grid upsampling
    tau: temperature for LSE approximation of max
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    
    # Check if zero
    if np.sum(h) < 1e-15:
        return float('inf')
    
    # Upsample to fine grid
    # x_fine in [-0.25, 0.25)
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    # Map fine grid points to step indices
    # Index i covers [-0.25 + i*W, -0.25 + (i+1)*W)
    # x + 0.25 is in [0, 0.5)
    # (x + 0.25) / W is in [0, N)
    cell_idx = np.floor((x_fine + 0.25) / W).astype(np.int32)
    cell_idx = np.clip(cell_idx, 0, N - 1)
    f_fine = h[cell_idx]
    
    dx = 0.5 / n_grid
    
    # Linear autoconvolution via FFT
    # f_fine is length n_grid. Convolution length 2*n_grid - 1.
    # Zero-pad to next power of 2 for speed.
    n_fft = 1 << (2 * n_grid - 1).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    
    # Scale by dx
    conv *= dx
    
    # LSE approximation of max(conv)
    # max(x) approx (1/tau) * log(sum(exp(tau * x)))
    # To avoid overflow, subtract max first
    max_conv = np.max(conv)
    # If max is very negative, exp(tau*conv) might underflow, but logsumexp handles it.
    # np.logaddexp.reduce is stable.
    # Let's use scipy.special.logsumexp if available, or manual stable version.
    # Manual: max + log(sum(exp(x - max)))
    scaled = tau * (conv - max_conv)
    sum_exp = np.sum(np.exp(scaled))
    lse_max = (1.0 / tau) * (np.log(sum_exp) + tau * max_conv)
    
    # Integral
    integral = np.sum(f_fine) * dx
    if integral < 1e-15:
        return float('inf')
        
    ratio = lse_max / (integral * integral)
    return ratio

def _compute_ratio_exact(h, n_grid=4096):
    """
    Compute the exact ratio: max(autoconv) / (int f)^2.
    Used for final evaluation and potentially as a local refinement objective.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.floor((x_fine + 0.25) / W).astype(np.int32)
    cell_idx = np.clip(cell_idx, 0, N - 1)
    f_fine = h[cell_idx]
    
    dx = 0.5 / n_grid
    
    n_fft = 1 << (2 * n_grid - 1).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    conv *= dx
    
    max_conv = np.max(conv)
    integral = np.sum(f_fine) * dx
    if integral < 1e-15:
        return float('inf')
        
    return max_conv / (integral * integral)

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Parameterization:
    # h_i = softplus( a_0 + sum_{k=1}^K a_k cos(2 pi k i / N) )
    # We assume symmetry, so no sine terms.
    # K controls the complexity.
    
    K = 20
    i = np.arange(N)
    
    # Precompute cosine basis
    basis = np.zeros((K, N))
    for k in range(1, K + 1):
        basis[k-1, :] = np.cos(2 * np.pi * k * i / N)
    
    def softplus(x):
        # Stable softplus
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    def get_heights(params):
        a0 = params[0]
        coeffs = params[1:1+K]
        z = np.full(N, a0)
        for k in range(K):
            z += coeffs[k] * basis[k]
        return softplus(z)
    
    # Objective function for optimization (uses LSE for smoothness)
    # We use a moderate tau for optimization to balance gradient accuracy and smoothness
    tau_opt = 1e-2 
    
    def objective(params):
        h = get_heights(params)
        return _compute_ratio_lse(h, n_grid=4096, tau=tau_opt)
    
    # Initial guess: Uniform
    # softplus(a0) = 1 => a0 = log(e-1) ~ 0.5413
    init_params = np.zeros(K + 1)
    init_params[0] = 0.5413
    
    best_h = get_heights(init_params)
    best_ratio = _compute_ratio_exact(best_h)
    
    # Optimization Loop
    rng = np.random.default_rng(42)
    
    # Try several initializations
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        if r == 0:
            p = init_params.copy()
        else:
            # Random initialization in Fourier space
            # Keep a0 near uniform level, randomize coefficients
            p = np.zeros(K + 1)
            p[0] = 0.5413 + rng.normal(0, 0.5)
            # Random coefficients, decay with frequency
            for k in range(1, K + 1):
                p[k] = rng.normal(0, 1.0 / (k + 1))
                
        try:
            # Use L-BFGS-B
            # maxiter should be sufficient for convergence
            res = minimize(objective, p, method='L-BFGS-B', 
                           options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < best_ratio:
                h_candidate = get_heights(res.x)
                # Verify with exact ratio
                exact_r = _compute_ratio_exact(h_candidate)
                if exact_r < best_ratio:
                    best_ratio = exact_r
                    best_h = h_candidate.copy()
                    
        except Exception as e:
            pass

    # Local Refinement
    # The Fourier representation might not capture sharp features.
    # We can try a few steps of coordinate descent on the full 600 variables.
    # However, 600 variables is high dimensional for random coordinate descent.
    # Instead, we can try to "snap" or refine the current best_h.
    
    # A better refinement: Use the current best_h as a starting point for a 
    # local optimization in the full space? No, too slow.
    
    # Let's try a simple local search on the Fourier coefficients with a tighter tau.
    # Or just run L-BFGS-B again on the best found parameters with a smaller tau.
    
    # Recover params from best_h? Inverse softplus and Fourier projection.
    # This is complex. Instead, just run L-BFGS-B on the best params found.
    # We didn't store the best params. Let's re-run the last successful one?
    # Better: Store best_params.
    
    # Let's restructure slightly to track best_params.
    # Since I already ran the loop, I can just do a final polish on the best_h 
    # by trying to fit it back to Fourier? No.
    
    # Alternative: Run L-BFGS-B on the best_h's "equivalent" Fourier coeffs?
    # Let's just do a final optimization run on the best_params from the last improvement.
    # I didn't save it. Let's just assume the global optimization is good enough.
    
    # One more trick: Run L-BFGS-B with a smaller tau (sharper max) starting from the best solution.
    # To do this, I need the params. Let's modify the loop to save best_params.
    
    # Re-running the logic to save params is safer.
    # But for now, the current best_h is the result of the global search.
    # Is it worth doing a local search on the 600 variables?
    # Gradient descent on 600 vars with FFT objective:
    # Each eval ~ 1ms. 100 steps * 600 dims (numerical grad) = 60000 evals = 60s.
    # Too slow.
    
    # So we rely on the Fourier optimization.
    
    return best_h.tolist()

# To make the code above work as intended with param tracking, I will rewrite construct() properly.

def construct():
    deadline = time.time() + 145.0
    
    K = 20
    i = np.arange(N)
    basis = np.zeros((K, N))
    for k in range(1, K + 1):
        basis[k-1, :] = np.cos(2 * np.pi * k * i / N)
        
    def softplus(x):
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
        
    def get_heights(params):
        a0 = params[0]
        coeffs = params[1:1+K]
        z = np.full(N, a0)
        for k in range(K):
            z += coeffs[k] * basis[k]
        return softplus(z)
        
    tau_opt = 1e-2
    def objective(params):
        h = get_heights(params)
        return _compute_ratio_lse(h, n_grid=4096, tau=tau_opt)
        
    init_params = np.zeros(K + 1)
    init_params[0] = 0.5413
    
    best_h = get_heights(init_params)
    best_ratio = _compute_ratio_exact(best_h)
    best_params = init_params.copy()
    
    rng = np.random.default_rng(42)
    n_restarts = 12
    
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        if r == 0:
            p = init_params.copy()
        else:
            p = np.zeros(K + 1)
            p[0] = 0.5413 + rng.normal(0, 0.5)
            for k in range(1, K + 1):
                p[k] = rng.normal(0, 1.0 / (k + 1))
                
        try:
            res = minimize(objective, p, method='L-BFGS-B', 
                           options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < best_ratio:
                h_candidate = get_heights(res.x)
                exact_r = _compute_ratio_exact(h_candidate)
                if exact_r < best_ratio:
                    best_ratio = exact_r
                    best_h = h_candidate.copy()
                    best_params = res.x.copy()
        except Exception:
            pass

    # Final Polish: Run L-BFGS-B on best_params with a smaller tau (sharper max approximation)
    # This helps converge to the true maximum rather than a smoothed one.
    if time.time() < deadline - 10.0:
        tau_fine = 1e-3
        def objective_fine(params):
            h = get_heights(params)
            return _compute_ratio_lse(h, n_grid=4096, tau=tau_fine)
            
        try:
            res_fine = minimize(objective_fine, best_params, method='L-BFGS-B', 
                                options={'maxiter': 100, 'gtol': 1e-9, 'disp': False})
            if res_fine.fun < best_ratio:
                h_candidate = get_heights(res_fine.x)
                exact_r = _compute_ratio_exact(h_candidate)
                if exact_r < best_ratio:
                    best_ratio = exact_r
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return best_h.tolist()
