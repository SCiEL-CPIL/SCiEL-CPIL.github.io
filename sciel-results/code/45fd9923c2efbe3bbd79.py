"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def autoconv_peak(h):
        """Compute the supremum norm of the autoconvolution (max over lags)."""
        n = len(h)
        # Linear autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k] for valid k
        # For lag k >= 0: sum_{i=0}^{n-1-k} h[i]*h[i+k]
        # For lag k < 0: sum_{i=0}^{n-1+k} h[i+k]*h[i] = same as lag |k| by symmetry
        # So we only need k = 0, 1, ..., n-1
        peaks = []
        for k in range(n):
            val = np.dot(h[:n-k], h[k:])
            peaks.append(val)
        return max(peaks)
    
    def ratio(h):
        """Compute the ratio: L2^2 / (L1 * sup_autoconv)."""
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        peak = autoconv_peak(h)
        if peak < 1e-12:
            return 0.0
        return l2_sq / (l1 * peak)
    
    def neg_ratio(h):
        return -ratio(h)
    
    # Initial guess: concentrated distribution
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try several initializations
    inits = []
    
    # Single spike
    for pos in [0, N//2, N-1]:
        h = np.zeros(N)
        h[pos] = 1.0
        inits.append(h)
    
    # Two adjacent spikes
    for pos in [0, N//2-1, N-2]:
        h = np.zeros(N)
        h[pos] = 0.5
        h[pos+1] = 0.5
        inits.append(h)
    
    # Triangular
    h = np.zeros(N)
    center = N // 2
    for i in range(N):
        h[i] = 1.0 - abs(i - center) / center if center > 0 else 1.0
    inits.append(h)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    h = np.exp(-5 * x**2)
    inits.append(h)
    
    # Uniform
    inits.append(np.ones(N))
    
    # Evaluate initializations
    for h0 in inits:
        if h0.sum() < 1e-12:
            continue
        s = ratio(h0)
        if s > best_score:
            best_score = s
            best_h = h0.copy()
    
    # L-BFGS-B optimization from best initialization
    constraints = []
    bounds = [(0.0, None)] * N
    
    # Run L-BFGS-B
    try:
        result = minimize(
            neg_ratio,
            best_h,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < neg_ratio(best_h):
            h_opt = np.maximum(result.x, 0.0)  # Ensure non-negativity
            s_opt = ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    except Exception:
        pass
    
    # Try a few more local optimizations from different starts
    for h0 in inits[:3]:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                neg_ratio,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if result.success or result.fun < neg_ratio(h0):
                h_opt = np.maximum(result.x, 0.0)
                s_opt = ratio(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Convert to list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
