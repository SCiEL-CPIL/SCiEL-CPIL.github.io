"""Optimized step function to minimize normalized max autoconvolution on [-1/4, 1/4]."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval length
h = (0.25 - (-0.25)) / N  # 0.5 / 600 = 1/1200

def _autoconvolve_max(a):
    """
    Compute the normalized maximum autoconvolution.
    Input a: array of heights.
    Returns: normalized max value.
    """
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def _softmax_max_obj_and_grad(a, beta=10.0):
    """
    Smooth surrogate for max autoconvolution using log-sum-exp.
    a: nonnegative array with sum 1
    Returns (value, gradient) where value approximates max(conv) and gradient is wrt a.
    """
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    
    # Log-sum-exp for smooth max
    max_c = np.max(conv)
    if max_c <= 0:
        return 0.0, np.zeros(n)
    shifted = conv / max_c
    log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    smooth_max = max_c * (1.0 + log_sum_exp)
    
    # Gradient of log-sum-exp wrt conv
    exp_terms = np.exp(beta * (shifted - 1.0))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient wrt a for each conv[k]
    # conv[k] = sum_i a[i] * a[i+k] for k >= 0 (with appropriate indexing)
    # For full linear convolution: conv[k] = sum_{i} a[i] * a[i - (k - (n-1))]... 
    # Actually for linear conv of a with a: conv[k] = sum_i a[i] * a[i + k - (n-1)]
    # Let's use the standard: conv = ifft(fft(a)*fft(a))
    # Gradient of conv[k] wrt a[j]:
    #   conv[k] = sum_m a[m] * a[m + k - (n-1)]
    #   d conv[k] / d a[j] = a[j + k - (n-1)] + a[j - k + (n-1)] (if in range)
    
    grad_a = np.zeros(n)
    for k in range(2*n-1):
        w = weights[k]
        if w < 1e-12:
            continue
        # conv[k] = sum_m a[m] * a[m + k - (n-1)]
        # d/d a[j] = a[j + k - (n-1)] if 0 <= j+k-(n-1) < n
        #           + a[j - k + (n-1)] if 0 <= j-k+(n-1) < n
        idx1 = k - (n - 1)
        idx2 = -idx1
        # First term: a[j + idx1] for j where 0 <= j+idx1 < n
        if idx1 >= 0:
            # j + idx1 in [0, n-1] => j in [0, n-1-idx1]
            if n - 1 - idx1 >= 0:
                grad_a[:n-idx1] += w * a[idx1:]
        else:
            # j + idx1 in [0, n-1] => j in [-idx1, n-1]
            if -idx1 < n:
                grad_a[-idx1:] += w * a[:n+idx1]
        # Second term: a[j + idx2] = a[j - idx1]
        # Same logic with idx2
        if idx2 >= 0:
            if n - 1 - idx2 >= 0:
                grad_a[:n-idx2] += w * a[idx2:]
        else:
            if -idx2 < n:
                grad_a[-idx2:] += w * a[:n+idx2]
    
    # Scale by max_c
    grad_a *= max_c
    
    return smooth_max, grad_a

def _logit_obj_smooth(b, beta=10.0):
    """Softmax parameterized smooth objective."""
    b_max = np.max(b)
    a = np.exp(b - b_max)
    s = np.sum(a)
    if s < 1e-300:
        return 1e10, np.zeros_like(b)
    a = a / s
    val, grad_a = _softmax_max_obj_and_grad(a, beta)
    dot = np.sum(grad_a * a)
    grad_b = a * (grad_a - dot)
    return val, grad_b

def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # Start with a reasonable initial guess
    x = np.linspace(-0.25, 0.25, N)
    
    # Try multiple initial profiles and keep the best
    profiles = []
    
    # Uniform
    a_uni = np.ones(N) / N
    profiles.append(a_uni)
    
    # Gaussian with different sigmas
    for sigma in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        a_g = np.exp(-x**2 / (2 * sigma**2))
        a_g = a_g / np.sum(a_g)
        profiles.append(a_g)
    
    # Raised cosine
    a_cos = 0.5 * (1 + np.cos(2 * np.pi * x / 0.5))
    a_cos = np.maximum(a_cos, 0)
    a_cos = a_cos / np.sum(a_cos)
    profiles.append(a_cos)
    
    # Triangular
    a_tri = np.maximum(1 - np.abs(x) / 0.25, 0)
    a_tri = a_tri / np.sum(a_tri)
    profiles.append(a_tri)
    
    # Parabolic
    a_par = np.maximum(1 - (x / 0.25)**2, 0)
    a_par = a_par / np.sum(a_par)
    profiles.append(a_par)
    
    # Find best initial profile
    best_a = None
    best_score = float('inf')
    for a_init in profiles:
        s = _autoconvolve_max(a_init)
        if s < best_score:
            best_score = s
            best_a = a_init.copy()
    
    # Optimization using L-BFGS-B with softmax reparameterization and smooth max
    def logit_obj(b):
        return _logit_obj_smooth(b, beta=15.0)
    
    # Optimize from the best initial profile
    b_init = np.log(best_a + 1e-15)
    
    try:
        result = minimize(logit_obj, b_init, jac=True, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 100})
        a_opt = np.exp(result.x)
        a_opt = a_opt / np.sum(a_opt)
        score_opt = _autoconvolve_max(a_opt)
        if score_opt < best_score:
            best_a = a_opt
            best_score = score_opt
    except Exception:
        pass
    
    # Try optimizing from other initial profiles if time permits
    for a_init in profiles:
        if time.time() > deadline - 30:
            break
        if np.allclose(a_init, best_a, atol=1e-6):
            continue
        
        b_init = np.log(a_init + 1e-15)
        try:
            result = minimize(logit_obj, b_init, jac=True, method='L-BFGS-B',
                              options={'maxiter': 200, 'gtol': 1e-7, 'maxls': 50})
            a_opt = np.exp(result.x)
            a_opt = a_opt / np.sum(a_opt)
            score_opt = _autoconvolve_max(a_opt)
            if score_opt < best_score:
                best_a = a_opt
                best_score = score_opt
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        b_polish = np.log(best_a + 1e-15)
        try:
            result = minimize(logit_obj, b_polish, jac=True, method='L-BFGS-B',
                              options={'maxiter': 100, 'gtol': 1e-10, 'maxls': 30})
            a_polish = np.exp(result.x)
            a_polish = a_polish / np.sum(a_polish)
            score_polish = _autoconvolve_max(a_polish)
            if score_polish < best_score:
                best_a = a_polish
                best_score = score_polish
        except Exception:
            pass
    
    return list(best_a)
