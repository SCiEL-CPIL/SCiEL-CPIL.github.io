import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute bin centers for parametric model evaluation
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = h[CELL]
    
    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def params_to_h(params):
    """
    Convert polynomial coefficients to step function heights.
    params: [a0_re, a0_im, a1_re, a1_im, ..., aK_re, aK_im]
    where h(x) = |sum_{k=0}^{K} a_k * exp(2*pi*i*k*x/0.5)|^2
    """
    K = len(params) // 2
    h = np.zeros(N)
    
    for i in range(N):
        x = BIN_CENTERS[i]
        # Compute sum of complex exponentials
        z = 0.0 + 0.0j
        for k in range(K + 1):
            a_re = params[2 * k]
            a_im = params[2 * k + 1]
            a = a_re + 1j * a_im
            theta = 2 * np.pi * k * x / 0.5
            z += a * np.exp(1j * theta)
        h[i] = np.abs(z) ** 2
    
    return h

def objective(params):
    h = params_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Number of polynomial terms (order K means K+1 coefficients)
    K = 3
    
    np.random.seed(42)
    
    # Define bounds for differential evolution
    # Each coefficient (real and imaginary) bounded in [-5, 5]
    bounds = [(-5.0, 5.0)] * (2 * (K + 1))
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.fun < best_val and np.isfinite(result.fun):
            h_candidate = params_to_h(result.x)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Try different orders
    for K_try in [2, 4]:
        if time.time() > deadline - 30.0:
            break
        
        bounds_try = [(-5.0, 5.0)] * (2 * (K_try + 1))
        try:
            result_try = differential_evolution(
                objective,
                bounds_try,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=42 + K_try,
                polish=True,
                init='sobol'
            )
            
            if result_try.fun < best_val and np.isfinite(result_try.fun):
                h_candidate = params_to_h(result_try.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        # Gradient-free local search: random perturbations
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb a random subset of bins
            n_perturb = np.random.randint(5, 20)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * 0.02
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    return list(best_h)
