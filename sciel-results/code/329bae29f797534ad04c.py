import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

def compute_ratio(h, n_grid=1000):
    """Compute max(autoconv) / l2_sq for step function h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
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

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    rng = np.random.default_rng(42)
    
    # Initialize best solution
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Strategy: Directly optimize the step heights using gradient-free methods
    # The objective is max_k (h*h)[k] / ||h||_2^2
    # We can use SLSQP or Nelder-Mead on the heights directly
    
    # Precompute the convolution matrix M such that (h*h)[k] = h^T M_k h
    # But that's expensive. Instead, we use FFT-based evaluation.
    
    # Better: Use a parameterization that ensures non-negativity
    # Let h = exp(y) for some y, or h = sigmoid(y) * scale
    
    # Actually, let's try a simpler approach:
    # 1. Start with various initial shapes
    # 2. Use Nelder-Mead or Powell to optimize
    
    # Initial guesses
    initial_guesses = []
    
    # Constant
    initial_guesses.append(np.ones(N))
    
    # Triangle peak
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    initial_guesses.append(np.maximum(0, 1 - 2*np.abs(x)))
    
    # Gaussian
    initial_guesses.append(np.exp(-20 * x**2))
    
    # Two peaks
    initial_guesses.append(np.maximum(0, 1 - 2*np.abs(x - 0.1)) + np.maximum(0, 1 - 2*np.abs(x + 0.1)))
    
    # Random smooth functions
    for _ in range(5):
        coeffs = rng.standard_normal(20)
        f = np.zeros(N)
        for k, c in enumerate(coeffs):
            f += c * np.cos(2*np.pi*k*x/0.5)
        f = np.maximum(0, f)
        if np.sum(f) > 0:
            initial_guesses.append(f)
    
    # Optimize each initial guess
    for i, h0 in enumerate(initial_guesses):
        if time.time() > deadline - 5:
            break
        
        # Normalize to avoid scaling issues
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.clip(h0, 0.0, None)
        if np.sum(h0) < 1e-10:
            continue
        h0 = h0 / np.sqrt(np.sum(h0**2))
        
        # Define objective function
        def objective(h_flat):
            h = np.clip(h_flat, 0.0, None)
            return compute_ratio(h, n_grid=500)
        
        # Try different optimizers
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(
                    objective,
                    h0,
                    method=method,
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5}
                )
                
                if res.fun < best_ratio:
                    h_opt = np.clip(res.x, 0.0, None)
                    r = compute_ratio(h_opt)
                    if r < best_ratio:
                        best_ratio = r
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final refinement with higher resolution
    if time.time() < deadline - 2:
        def objective_refine(h_flat):
            h = np.clip(h_flat, 0.0, None)
            return compute_ratio(h, n_grid=2000)
        
        try:
            res = minimize(
                objective_refine,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            
            if res.fun < best_ratio:
                h_opt = np.clip(res.x, 0.0, None)
                r = compute_ratio(h_opt)
                if r < best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h.tolist()
