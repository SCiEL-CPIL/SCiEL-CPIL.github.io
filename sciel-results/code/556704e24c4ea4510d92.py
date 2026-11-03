import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# Precompute fine grid for fast evaluation
# Use a grid finer than the step function to capture the autoconvolution peak accurately
# The step function is constant on intervals of width W.
# The autoconvolution of a step function is a piecewise quadratic function.
# Its maximum will occur either at a peak of the continuous approximation or at a breakpoint.
# Using a fine grid (e.g., 12000 points) is a safe and fast way to approximate the max.
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to step indices
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Expand to fine grid
    f_fine = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    max_conv = np.max(conv)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_conv / l2_sq

def params_to_h(params):
    """
    Convert parameters to a step function.
    params: [amp1, center1, width1, amp2, center2, width2, ...]
    """
    h = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    n_gaussians = len(params) // 3
    for i in range(n_gaussians):
        amp = params[3*i]
        center = params[3*i+1]
        width = params[3*i+2]
        # Ensure width is positive
        width = max(abs(width), 1e-3)
        h += amp * np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    return h

def objective(params):
    h = params_to_h(params)
    # Ensure non-negativity for the ratio calculation, 
    # but the Gaussian sum can be negative if amps are negative.
    # The problem requires a nonnegative step function.
    # We can either constrain amps to be positive or clip h.
    # Clipping h is non-differentiable. 
    # Instead, let's assume we are summing positive Gaussians.
    # But optimization might try negative amps to create dips.
    # If we clip, the gradient is zero where h<0, which is fine for L-BFGS-B mostly.
    # However, a better way is to model h = |sum| or just sum of positive Gaussians.
    # Let's stick to sum of Gaussians with unconstrained amps, then clip for evaluation.
    # This is a heuristic.
    h_clipped = np.clip(h, 0, None)
    val = ratio_from_h(h_clipped)
    # Add a small penalty if the function is zero
    if np.sum(h_clipped) < 1e-10:
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: a few Gaussians
    # Try different initial configurations
    best_params = None
    best_val = float('inf')
    
    # Configuration 1: Single central Gaussian
    # Configuration 2: Two asymmetric Gaussians
    # Configuration 3: Three Gaussians
    
    configs = [
        # [amp, center, width]
        [1.0, 0.0, 0.1],
        [1.0, -0.1, 0.1, 1.0, 0.1, 0.1],
        [1.0, -0.15, 0.1, 0.5, 0.0, 0.2, 1.0, 0.15, 0.1],
        [1.0, 0.0, 0.05, 1.0, -0.1, 0.05, 1.0, 0.1, 0.05],
    ]
    
    for config in configs:
        if time.time() > deadline:
            break
            
        # Randomize initial parameters slightly
        params = np.array(config, dtype=np.float64)
        # Add noise
        params += np.random.randn(len(params)) * 0.05
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # Also try a broader random search with optimization
    np.random.seed(42)
    n_random_starts = 50
    
    for i in range(n_random_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Random number of Gaussians: 1 to 4
        n_g = np.random.randint(1, 5)
        params = np.zeros(3 * n_g)
        for j in range(n_g):
            params[3*j] = np.random.uniform(0.1, 2.0) # Amp
            params[3*j+1] = np.random.uniform(-0.2, 0.2) # Center
            params[3*j+2] = np.random.uniform(0.02, 0.15) # Width
            
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'maxfun': 100, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final polish: use a higher precision grid for the best candidate?
    # The ratio_from_h uses FINE_N=12000. This should be sufficient.
    # We can try to refine the parameters of the best solution further.
    
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Convert best params to h
    if best_params is not None:
        best_h = params_to_h(best_params)
    else:
        # Fallback: uniform
        best_h = np.ones(N)
        
    best_h = np.clip(best_h, 0, None)
    
    # Verify the score
    final_val = ratio_from_h(best_h)
    
    return list(best_h)
