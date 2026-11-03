import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
DOMAIN_LEN = 0.5
X_STEP = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast evaluation on a fine grid
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = DOMAIN_LEN / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h_step):
    """
    Compute the normalized maximum of the autoconvolution.
    h_step: array of N non-negative step heights.
    """
    if np.sum(h_step) < 1e-15:
        return float('inf')
    
    # Expand to fine grid
    h_fine = h_step[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(h_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    max_conv = np.max(conv)
    l2_sq = np.sum(h_fine ** 2) * DX
    
    if l2_sq < 1e-15:
        return float('inf')
        
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Strategy:
    # 1. Start with a Gaussian-like bump (which is known to minimize the L-infinity norm of convolution for fixed L2 norm in some contexts, though here we have the L1 constraint via non-negativity and the specific domain).
    #    Actually, the "optimal" shape for minimizing the peak of the autoconvolution given a fixed integral (or mass) and support is related to the prolate spheroidal wave functions or simply a Gaussian.
    #    Let's initialize with a Gaussian centered at 0.
    
    x = X_STEP
    sigma = 0.1
    h_init = np.exp(-x**2 / (2 * sigma**2))
    # Normalize so that sum(h) is some reasonable value, or just keep it.
    # The ratio is scale-invariant: (c^2 * conv_max) / (c^2 * l2_sq) = conv_max / l2_sq.
    # So normalization doesn't change the score, only numerical stability.
    # Let's normalize to sum=1 for stability.
    h_init = h_init / np.sum(h_init)
    
    best_h = h_init.copy()
    best_val = ratio_from_h(best_h)
    
    # Optimization Phase 1: L-BFGS-B on the raw heights
    # We need to handle the non-negativity constraint. L-BFGS-B supports bounds.
    bounds = [(0, None)] * N
    
    def objective(h):
        h = np.clip(h, 0, None)
        return ratio_from_h(h)
    
    # Run L-BFGS-B from the Gaussian start
    try:
        res = minimize(
            objective, 
            best_h, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
    except Exception:
        pass
    
    # Optimization Phase 2: Multi-start with different widths and shapes
    # Try different sigmas for the Gaussian initialization
    sigmas = [0.05, 0.08, 0.12, 0.15, 0.2]
    
    for sigma in sigmas:
        if time.time() > deadline - 20.0:
            break
            
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        h_gauss = h_gauss / np.sum(h_gauss)
        
        try:
            res = minimize(
                objective, 
                h_gauss, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
            
    # Optimization Phase 3: Nelder-Mead on the best solution found so far
    # N-M can sometimes escape local minima that gradient methods get stuck in,
    # especially with the non-smoothness introduced by clipping (though we clip in objective,
    # the gradient is computed on the unclipped values, which can be problematic if we go negative.
    # However, L-BFGS-B keeps us in bounds. N-M might step out of bounds.
    # We rely on the clipping in the objective function to handle out-of-bounds steps,
    # effectively making the objective flat for negative values. This is a common trick.
    
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False, 'initial_simplex': None}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
            
    # Final Polish: Another L-BFGS-B run with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
            
    best_h = np.clip(best_h, 0, None)
    
    # Ensure the result is a list of floats as required
    return list(best_h)
