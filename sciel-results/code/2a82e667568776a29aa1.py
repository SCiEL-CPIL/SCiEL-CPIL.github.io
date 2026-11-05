import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s safety margin

def _compute_value(h):
    h = np.asarray(h)
    s = np.sum(h)
    if np.isclose(s, 0):
        return np.inf
    conv = fftconvolve(h, h, mode='full')
    max_conv = np.max(conv)
    val = abs(2 * N * max_conv / (s * s))
    return val

def _lse_objective(h_flat, tau=10.0):
    # h_flat is the vector to optimize
    h = h_flat.reshape(-1)
    # Normalize to unit norm to handle scale invariance of the direction
    norm = np.linalg.norm(h)
    if norm < 1e-10:
        return 1e10
    h_norm = h / norm
    
    conv = fftconvolve(h_norm, h_norm, mode='full')
    
    # LSE approximation of max
    # Use stable log-sum-exp
    max_conv = np.max(conv)
    if max_conv > 0:
        lse = max_conv + np.log(np.sum(np.exp(tau * (conv - max_conv)))) / tau
    else:
        # If max is negative, exp(tau * conv) might underflow, but max is small
        # Actually if max is negative, exp(tau * conv) is very small.
        # The log sum will be dominated by the largest term.
        # lse = (1/tau) * log(sum(exp(tau * conv)))
        # If tau * conv is very negative, this is problematic.
        # But we are minimizing, so we want it to be negative.
        # Let's just compute it carefully.
        shifted = tau * (conv - max_conv)
        # shifted <= 0
        sum_exp = np.sum(np.exp(shifted))
        lse = max_conv + np.log(sum_exp) / tau
        
    return lse

def _gradient_lse(h_flat, tau=10.0):
    h = h_flat.reshape(-1)
    norm = np.linalg.norm(h)
    if norm < 1e-10:
        return np.zeros_like(h)
    h_norm = h / norm
    
    conv = fftconvolve(h_norm, h_norm, mode='full')
    max_conv = np.max(conv)
    shifted = tau * (conv - max_conv)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # d(lse)/d(conv_i) = tau * exp(tau * (conv_i - max_conv)) / sum_exp
    d_lse_d_conv = tau * exp_shifted / sum_exp
    
    # d(conv_i)/d(h_norm_j)
    # conv is full convolution of length 2N-1
    # conv[k] = sum_j h_norm[j] h_norm[k-j] (with indices adjusted)
    # Using FFT gradient is complex. Let's use numerical gradient or finite differences if needed?
    # L-BFGS-B can do finite differences, but it's slow.
    # Let's try to derive it or use a simpler objective.
    # Actually, minimizing LSE of conv is equivalent to minimizing the "soft" max.
    # We can just use numerical gradient if it's fast enough.
    # N=400, conv length 799.
    # Finite difference gradient: 400 evaluations of objective.
    # Objective involves FFT conv (fast) and LSE (fast).
    # 400 * (FFT + LSE) is very fast.
    # So we can let L-BFGS-B use finite differences.
    return None # Signal to use finite differences

def construct():
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    np.random.seed(42)
    
    num_starts = 50
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Random initialization
        h0 = np.random.randn(N)
        
        # Try to minimize LSE of conv
        # We use L-BFGS-B with finite differences
        res = minimize(_lse_objective, h0, method='L-BFGS-B', 
                       options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False})
        
        if res.success or res.status == 0:
            h_opt = res.x
            # Normalize h_opt to have sum 1? No, value is scale invariant.
            # But we need to compute the value correctly.
            val = _compute_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        
        # Also try a few other initializations:
        # 1. Sign-reversing
        if i % 5 == 1:
            h0 = np.where(np.arange(N) % 2 == 0, 1, -1)
            res = minimize(_lse_objective, h0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False})
            if res.success or res.status == 0:
                h_opt = res.x
                val = _compute_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        
        # 2. Sine wave
        if i % 5 == 2:
            h0 = np.sin(np.linspace(0, 10*np.pi, N))
            res = minimize(_lse_objective, h0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False})
            if res.success or res.status == 0:
                h_opt = res.x
                val = _compute_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()

    # Final polish: try to reduce the max conv further
    # If best_val is still high, try more iterations on the best_h
    if time.time() - start_time < TIME_LIMIT - 5:
        res = minimize(_lse_objective, best_h, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False})
        if res.success or res.status == 0:
            h_opt = res.x
            val = _compute_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()

    return best_h
