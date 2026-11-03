import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution using L1 normalization."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = h[CELL]
    
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    max_conv = np.max(conv)
    
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def params_to_h(params):
    # Same as current, using BIN_CENTERS for speed
    K = len(params) // 3
    h = np.zeros(N)
    # Precompute centers
    BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W
    for i in range(K):
        a_raw = params[3*i]
        x_c = params[3*i + 1]
        s_raw = params[3*i + 2]
        
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    return h

def objective(params):
    h = params_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    deadline = time.time() + 145.0
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # 1. Parametric Optimization (Gaussian Superposition)
    K = 5
    np.random.seed(42)
    inits = []
    
    # ... (generate inits as in current) ...
    
    for p0 in inits:
        if time.time() > deadline - 60: # Leave time for direct opt
            break
        try:
            res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < best_val:
                h_candidate = params_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # 2. Direct Optimization with Soft-Max Surrogate
    if time.time() < deadline - 10:
        # We optimize h directly
        # Objective: SoftMax(conv) / (sum(h)*W)^2
        beta = 200.0 # Smoothness parameter
        
        def obj_direct_smooth(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * W) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            # Soft Max
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            # log(sum(exp(conv - m))) + m
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
            
        bounds = [(0.0, None)] * N
        try:
            res_direct = minimize(
                obj_direct_smooth,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = ratio_from_h(h_candidate) # Use true max for scoring
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
                
        except Exception:
            pass
            
    return list(best_h)
