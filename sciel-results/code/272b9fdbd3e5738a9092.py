"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def _autoconv_max(h):
    """Compute max of autoconvolution for step function heights h."""
    # Discrete autoconvolution: c[k] = sum_i h[i]*h[i+k] for valid i
    # Using FFT for efficiency
    n = len(h)
    size = 2 * n
    fft_h = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(fft_h * fft_h, n=size)
    # The autoconvolution (not cross) - we need sum h[i]*h[i+k]
    # This is the linear convolution of h with reversed h, but for autoconv of nonneg
    # Actually for autoconvolution of step function, the value at lag k is sum_i h[i]h[i+k]
    # Let's compute it directly via correlation
    # c[k] = sum_{i} h[i] * h[i+k] for k >= 0, and symmetric for k < 0
    # Use FFT-based approach
    # Linear convolution of h with h gives sum h[i]h[j] where i+j = k, not what we want
    # We want sum h[i]h[i+k], which is cross-correlation
    # np.correlate(h, h, mode='full') gives sum h[i]h[i+k] 
    corr = np.correlate(h, h, mode='full')
    return np.max(corr)

def _objective(x, w):
    """Objective: max(autoconv) / (int f)^2, where int f = w * sum(x)"""
    h = np.maximum(x, 0)  # enforce nonnegativity
    if np.sum(h) < 1e-12:
        return 1e10
    integral = w * np.sum(h)
    max_conv = _autoconv_max(h)
    return max_conv / (integral * integral)

def construct():
    deadline = time.time() + 120  # 120s wall clock limit, use 120s budget
    w = 0.5 / N  # interval width, domain [-1/4, 1/4] has length 1/2
    
    best_h = np.ones(N)
    best_val = _objective(best_h, w)
    
    # Try several initial points and local optimizations
    rng = np.random.default_rng(42)
    
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        
        # Different initializations
        if trial == 0:
            h0 = np.ones(N)
        elif trial == 1:
            h0 = np.exp(-np.linspace(-3, 3, N)**2)
        elif trial == 2:
            h0 = np.ones(N) * np.exp(-np.linspace(0, 2, N))
        elif trial == 3:
            h0 = np.maximum(np.sin(np.linspace(0, np.pi, N)), 0)
        elif trial == 4:
            h0 = np.maximum(np.cos(np.linspace(0, np.pi, N)), 0)
        elif trial == 5:
            # Two bumps
            h0 = np.zeros(N)
            h0[:N//4] = 1.0
            h0[3*N//4:] = 1.0
        elif trial == 6:
            # Central bump
            h0 = np.zeros(N)
            h0[N//4:3*N//4] = 1.0
        elif trial == 7:
            h0 = rng.random(N) + 0.1
        elif trial == 8:
            h0 = np.ones(N) * (1 + 0.5*np.sin(np.linspace(0, 4*np.pi, N)))
            h0 = np.maximum(h0, 0)
        elif trial == 9:
            h0 = np.ones(N) * (1 + 0.3*np.cos(np.linspace(0, 2*np.pi, N)))
            h0 = np.maximum(h0, 0)
        else:
            # Perturb best
            h0 = best_h * (1 + 0.1 * rng.standard_normal(N))
            h0 = np.maximum(h0, 0)
        
        # Normalize to avoid scaling issues
        h0 = h0 / np.sum(h0) * N  # keep similar scale
        
        # Use L-BFGS-B with bounds
        bounds = [(0, None)] * N
        
        result = minimize(
            _objective, h0, args=(w,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_val:
            h_new = np.maximum(result.x, 0)
            val = _objective(h_new, w)
            if val < best_val:
                best_val = val
                best_h = h_new
    
    # Polish with tighter tolerance
    if time.time() < deadline - 10:
        result = minimize(
            _objective, best_h, args=(w,),
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if result.fun < best_val:
            best_h = np.maximum(result.x, 0)
            best_val = _objective(best_h, w)
    
    return list(best_h)
# EVOLVE-BLOCK-END
