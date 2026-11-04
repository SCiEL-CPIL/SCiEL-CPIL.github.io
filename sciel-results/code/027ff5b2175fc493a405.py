import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

def _compute_ratio_fast(h, n_grid=4000):
    """Compute max(autoconv) / l2_sq for step function h efficiently."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]

    # Linear autoconvolution via FFT (zero-padded)
    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)

    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _compute_ratio(h, n_grid=2000):
    """Wrapper for compatibility."""
    return _compute_ratio_fast(h, n_grid)

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with uniform
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_ratio = _compute_ratio_fast(best_h)
    
    # Parameterization: h = softplus(z)
    # z = a0 * ones + sum_{k=1}^K a_k cos(2pi k i/N) + b_k sin(2pi k i/N)
    K = 15 # Number of Fourier modes
    i = np.arange(N)
    # Precompute basis functions
    basis = np.zeros((K, N))
    for k in range(1, K + 1):
        basis[k-1, :] = np.cos(2 * np.pi * k * i / N)
    # We can skip sine terms if we assume symmetry, or include them.
    # Let's include both for generality, but symmetry might be optimal.
    # Let's try symmetric first (only cosines) to reduce params.
    
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def get_heights(params):
        a0 = params[0]
        coeffs = params[1:1+K]
        z = np.full(N, a0)
        for k in range(K):
            z += coeffs[k] * basis[k]
        return softplus(z)

    def objective(params):
        h = get_heights(params)
        # Ensure non-negative (softplus guarantees > 0, but clip for safety)
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            return 1e10
        return _compute_ratio_fast(h)
    
    # Initial guess: Uniform
    # We want h ~ 1.
    # softplus(z) ~ 1 => z ~ log(e-1) ~ 0.54
    # Let's just set initial params to produce ~1
    # If coeffs=0, h = softplus(a0).
    # softplus(0.54) ~ 1.
    init_params = np.zeros(K + 1)
    init_params[0] = 0.54
    
    # Optimize
    # Use L-BFGS-B
    # We need a gradient. Numerical gradient is slow for 16 params? No, 16 is small.
    # But objective is called many times.
    # Let's try analytical gradient if possible, or just let scipy do it.
    # SciPy's L-BFGS-B uses numerical gradient by default if not provided.
    # With 16 parameters, numerical gradient is 16 evaluations per step.
    # Each evaluation is ~1ms (FFT). 16ms per step. 100 steps = 1.6s.
    # This is very fast.
    
    try:
        result = minimize(objective, init_params, method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
        
        if result.success or result.fun < best_ratio:
            best_h = get_heights(result.x)
            best_ratio = result.fun
    except Exception:
        pass
    
    # Try a few different initializations to avoid local minima
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        # Random small perturbations in Fourier space
        init_params = np.zeros(K + 1)
        init_params[0] = 0.54
        # Add small random noise to coefficients
        noise = rng.normal(0, 0.1, size=K)
        init_params[1:1+K] = noise
        
        try:
            result = minimize(objective, init_params, method='L-BFGS-B', 
                              options={'maxiter': 100, 'gtol': 1e-8, 'disp': False})
            if result.fun < best_ratio:
                best_h = get_heights(result.x)
                best_ratio = result.fun
        except Exception:
            pass

    # Local refinement on the full 600 variables using coordinate descent
    # This can fine-tune the result from the Fourier approximation
    h = best_h.copy()
    lr = 0.01
    step_size = max(0.01, np.max(h) * 0.01)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        n_trials = 20
        idxs = rng.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            h[idx] = h_old - step_size
            r_minus = _compute_ratio_fast(h)
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
        else:
            h = best_h if False else h # h is already updated in place

    # Final check
    final_r = _compute_ratio_fast(best_h)
    if final_r < best_ratio:
        best_ratio = final_r
        # Update best_h if needed, but best_h is already the one used for final_r?
        # In the loop above, h is modified. best_h was the Fourier result.
        # We should track the best h found in the refinement.
        # Let's re-run the logic to ensure we keep the best h.
    
    # Re-structure refinement to track best_h properly
    h_refine = best_h.copy()
    best_ratio_refine = best_ratio
    step_size = max(0.01, np.max(h_refine) * 0.01)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        n_trials = 20
        idxs = rng.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h_refine[idx]
            h_refine[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h_refine)
            h_refine[idx] = h_old - step_size
            r_minus = _compute_ratio_fast(h_refine)
            h_refine[idx] = h_old
            
            if r_plus < best_ratio_refine:
                h_refine[idx] = h_old + step_size
                best_ratio_refine = r_plus
                improved = True
            elif r_minus < best_ratio_refine:
                h_refine[idx] = h_old - step_size
                best_ratio_refine = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break

    if best_ratio_refine < best_ratio:
        best_h = h_refine
        best_ratio = best_ratio_refine

    return best_h.tolist()
