import numpy as np
import time
from scipy.optimize import minimize

N = 400

def compute_value(h):
    """Compute the task value."""
    h_sum = np.sum(h)
    if h_sum == 0 or not np.isfinite(h_sum):
        return float('inf')
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (h_sum ** 2))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2*N-1]

def _smooth_max_value(h, tau=0.1):
    """Smooth surrogate for max(autoconv) using log-sum-exp."""
    conv = _fft_convolve(h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    # Soft max via log-sum-exp
    m = np.max(conv)
    log_sum_exp = m + np.log(np.sum(np.exp((conv - m) / tau)))
    return abs(2.0 * N * log_sum_exp / (s * s))

def _build_nonharmonic(params, n_modes):
    """
    Build h from non-harmonic sinusoid parameters.
    params layout: [offset, A0, f0, p0, A1, f1, p1, ..., A_{K-1}, f_{K-1}, p_{K-1}]
    h[n] = offset + sum_k A_k * cos(2*pi*f_k*n + p_k)
    """
    n = np.arange(N, dtype=np.float64)
    offset = params[0]
    h = np.full(N, offset, dtype=np.float64)
    for k in range(n_modes):
        A_k = params[1 + 3*k]
        f_k = params[2 + 3*k]
        p_k = params[3 + 3*k]
        h += A_k * np.cos(2.0 * np.pi * f_k * n + p_k)
    return h

def _nonharmonic_objective(params, n_modes, tau=0.1):
    """Smooth objective for non-harmonic sinusoid parameters."""
    h = _build_nonharmonic(params, n_modes)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    conv = _fft_convolve(h)
    m = np.max(conv)
    log_sum_exp = m + np.log(np.sum(np.exp((conv - m) / tau)))
    return abs(2.0 * N * log_sum_exp / (s * s))

def _nonharmonic_grad(params, n_modes, tau=0.1):
    """
    Numerical gradient of smooth objective w.r.t. non-harmonic params.
    Uses central differences for accuracy.
    """
    eps = 1e-5
    grad = np.zeros_like(params)
    f0 = _nonharmonic_objective(params, n_modes, tau)
    for i in range(len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += eps
        params_m[i] -= eps
        fp = _nonharmonic_objective(params_p, n_modes, tau)
        fm = _nonharmonic_objective(params_m, n_modes, tau)
        grad[i] = (fp - fm) / (2 * eps)
    return grad

def _optimize_nonharmonic(n_modes, time_limit=30.0, seed=42):
    """
    Optimize non-harmonic sinusoid parameters using gradient-based methods.
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Parameter bounds
    # offset: [-1, 1]
    # A_k: [0, 1.5]
    # f_k: [0.01, 2.0] (cycles over N samples)
    # p_k: [-pi, pi]
    
    best_params = None
    best_val = 1e10
    
    # Multiple random restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start > time_limit - 5:
            break
        
        # Initialize parameters
        params = np.zeros(1 + 3 * n_modes)
        params[0] = rng.uniform(-0.5, 0.5)  # offset
        for k in range(n_modes):
            params[1 + 3*k] = rng.uniform(0.1, 1.0)      # A_k
            params[2 + 3*k] = rng.uniform(0.05, 1.5)     # f_k
            params[3 + 3*k] = rng.uniform(-np.pi, np.pi) # p_k
        
        # Gradient descent with adaptive step size
        lr = 0.01
        for it in range(500):
            if time.time() - start > time_limit - 5:
                break
            
            val = _nonharmonic_objective(params, n_modes, tau=0.1)
            if val < best_val:
                best_val = val
                best_params = params.copy()
            
            # Compute gradient
            try:
                g = _nonharmonic_grad(params, n_modes, tau=0.1)
            except:
                break
            
            # Check for NaN/inf
            if not np.all(np.isfinite(g)):
                break
            
            # Adaptive learning rate
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            
            # Line search: try several step sizes
            best_step_val = val
            best_step_params = params.copy()
            for lr_try in [lr, lr*0.5, lr*0.25, lr*0.1, lr*0.05]:
                params_try = params - lr_try * g
                val_try = _nonharmonic_objective(params_try, n_modes, tau=0.1)
                if val_try < best_step_val:
                    best_step_val = val_try
                    best_step_params = params_try
                    lr = lr_try  # Keep successful step size
                    break
            else:
                # If no improvement, reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
                continue
            
            params = best_step_params
        
        # Final local polish with L-BFGS-B
        if time.time() - start < time_limit - 10:
            try:
                res = minimize(
                    lambda p: _nonharmonic_objective(p, n_modes, tau=0.05),
                    best_params if best_params is not None else params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except:
                pass
    
    if best_params is None:
        return None, 1e10
    
    h = _build_nonharmonic(best_params, n_modes)
    actual_val = compute_value(h)
    return h, actual_val

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0
    
    # Try non-harmonic sinusoid optimization with different numbers of modes
    mode_configs = [
        (3, 40),   # 3 modes, 40s
        (5, 40),   # 5 modes, 40s
        (4, 35),   # 4 modes, 35s
        (6, 30),   # 6 modes, 30s
    ]
    
    for n_modes, t_alloc in mode_configs:
        if time.time() - start_time > time_limit - 5:
            break
        
        h_try, val_try = _optimize_nonharmonic(n_modes, time_limit=t_alloc, seed=42)
        
        if h_try is not None and val_try < best_val:
            best_val = val_try
            best_h = h_try.copy()
        
        # Also try a different seed for diversity
        if time.time() - start_time < time_limit - 10:
            h_try2, val_try2 = _optimize_nonharmonic(n_modes, time_limit=min(20, t_alloc//2), seed=123)
            if h_try2 is not None and val_try2 < best_val:
                best_val = val_try2
                best_h = h_try2.copy()
    
    # Final polish: direct gradient descent on h with smooth objective
    if time.time() - start_time < time_limit - 10:
        h = best_h.copy()
        h_sum = np.sum(h)
        if abs(h_sum) > 1e-10:
            h = h / h_sum  # normalize to sum 1
        
        tau = 0.1
        lr = 0.005
        for it in range(300):
            if time.time() - start_time > time_limit - 5:
                break
            
            conv = _fft_convolve(h)
            m = np.max(conv)
            log_sum_exp = m + np.log(np.sum(np.exp((conv - m) / tau)))
            val = abs(2.0 * N * log_sum_exp)  # sum is 1
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Compute gradient of smooth max autoconv w.r.t. h
            # d/dh_j of max_k conv[k] where conv[k] = sum_i h[i]*h[i+k-N]
            # Using soft max: grad = (1/tau) * exp((conv - m)/tau) / sum(exp(...)) * dconv/dh
            weights = np.exp((conv - m) / tau)
            weights = weights / np.sum(weights)
            
            # dconv/dh_j = sum over k of weights[k] * dconv[k]/dh[j]
            # conv[k] = sum_i h[i] * h[i + k - N] for k in [0, 2N-2]
            # dconv[k]/dh[j] = h[j + k - N] if 0 <= j+k-N < N, plus h[j - k + N] if 0 <= j-k+N < N
            # Actually conv is full convolution: conv[k] = sum_i h[i] * h[i + k - (N-1)]
            # Let's use the standard: for full conv, conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
            # dconv[k]/dh[j] = h[k-j] if 0 <= k-j < N, plus h[j-k+N-1]... this is getting complex.
            # Use numerical gradient instead for simplicity and correctness
            
            g = np.zeros(N)
            eps = 1e-5
            for j in range(N):
                h_p = h.copy()
                h_m = h.copy()
                h_p[j] += eps
                h_m[j] -= eps
                cp = _fft_convolve(h_p)
                cm = _fft_convolve(h_m)
                mp = np.max(cp)
                mm = np.max(cm)
                lse_p = mp + np.log(np.sum(np.exp((cp - mp) / tau)))
                lse_m = mm + np.log(np.sum(np.exp((cm - mm) / tau)))
                g[j] = (lse_p - lse_m) / (2 * eps)
            
            # Subtract mean to maintain zero-mean constraint (sum stays 1)
            g = g - np.mean(g)
            
            h = h - lr * g
            h = h - (np.sum(h) - 1) / N  # Renormalize
            
            lr *= 0.995
    
    return best_h
