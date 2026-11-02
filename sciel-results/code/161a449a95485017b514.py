"""Minimize normalized max autoconvolution of nonnegative step function on [-1/4, 1/4]."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def _normalized_max_autoconv(h):
    """Normalized max autoconvolution, scale-invariant."""
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
    max_conv = np.max(conv[:N])
    return max_conv / (s * s)

def _smooth_max_autoconv(h, temperature=0.05):
    """Smooth surrogate using log-sum-exp."""
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
    conv_lags = np.maximum(conv[:N], 1e-15)
    log_sum_exp = temperature * np.log(np.sum(np.exp(conv_lags / temperature)))
    return log_sum_exp / (s * s)

def construct():
    deadline = time.time() + 140
    
    # Block function: 1 on first half, 0 on second
    h0 = np.zeros(N)
    h0[:N // 2] = 1.0
    
    best_h = h0.copy()
    best_val = _normalized_max_autoconv(h0)
    
    # Try multiple initial configurations
    inits = [
        np.concatenate([np.ones(N // 2), np.zeros(N // 2)]),
        np.concatenate([np.zeros(N // 4), np.ones(N // 2), np.zeros(N // 4)]),
        np.concatenate([np.ones(N // 3), np.zeros(N // 3), np.ones(N // 3)]),
        np.concatenate([np.ones(N // 4), np.zeros(N // 2), np.ones(N // 4)]),
    ]
    
    for init in inits:
        if time.time() > deadline - 40:
            break
        val = _normalized_max_autoconv(init)
        if val < best_val:
            best_val = val
            best_h = init.copy()
    
    # Optimize with smooth max
    try:
        res = minimize(
            lambda h: _smooth_max_autoconv(h, 0.05), best_h,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < best_val:
            h_opt = np.maximum(res.x, 0)
            val = _normalized_max_autoconv(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Sharper temperature
    if time.time() < deadline - 20:
        try:
            res = minimize(
                lambda h: _smooth_max_autoconv(h, 0.01), best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                h_opt = np.maximum(res.x, 0)
                val = _normalized_max_autoconv(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with exact
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _normalized_max_autoconv, best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.success:
                h_opt = np.maximum(res.x, 0)
                val = _normalized_max_autoconv(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
