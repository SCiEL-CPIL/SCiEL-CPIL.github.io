"""
Minimize the normalized maximum of autoconvolution of a nonnegative step
function on [-1/4, 1/4] with 600 equal intervals.

Strategy: represent the step function as a linear combination of a small
number of DCT basis modes with non-negative coefficients. This reduces
the optimization dimension from 600 to ~K, enabling a global search
(Nelder-Mead) to find a sparse spectral representation that flattens
the autoconvolution peak more effectively than dense local methods.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
# DCT basis modes to use (low frequency modes capture smooth shapes)
K = 20

# Precompute DCT basis matrix: B[i, k] = cos(pi * k * (i + 0.5) / N)
# This is the DCT-II basis.
def _build_basis():
    i = np.arange(N)
    k = np.arange(K)
    # DCT-II: cos(pi * k * (2*i + 1) / (2*N))
    B = np.cos(np.pi * np.outer(k, 2 * i + 1) / (2 * N))
    return B

B = _build_basis()

def _autoconv_max(coeffs):
    """Compute the maximum of the autoconvolution of the step function
    defined by DCT coefficients `coeffs` (non-negative)."""
    # Reconstruct step function
    f = coeffs @ B  # shape (N,)
    # Ensure non-negative
    f = np.maximum(f, 0.0)
    
    # Autoconvolution via FFT
    # The autoconvolution of f with itself: (f * f)(t) = integral f(s) f(t-s) ds
    # For discrete step function on N intervals:
    # autoconv[i] = sum_j f[j] * f[i - j]  (with appropriate bounds)
    # Using FFT for efficiency
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    autoconv = np.fft.irfft(F * F, n=n_fft)
    
    # The autoconvolution is defined on [-N, N) intervals
    # We need the maximum over the full range
    # But the step function is supported on [-1/4, 1/4], so the autoconv
    # is supported on [-1/2, 1/2]. The discrete autoconv has indices 0..N-1
    # corresponding to the convolution result.
    
    # Actually, let's think more carefully. The step function f has N values
    # on [-1/4, 1/4]. The autoconvolution (f * f)(t) = int f(s) f(t-s) ds.
    # Since f is supported on [-1/4, 1/4], f * f is supported on [-1/2, 1/2].
    
    # The discrete convolution: if we pad f to length 2N and do circular conv,
    # or use linear convolution.
    
    # Let's use linear convolution via FFT
    n_lin = 2 * N - 1
    F_lin = np.fft.rfft(f, n=n_lin)
    autoconv_lin = np.fft.irfft(F_lin * F_lin, n=n_lin)
    
    # The autoconvolution values: autoconv_lin[k] corresponds to sum_j f[j] f[k-j]
    # for k = 0, 1, ..., 2N-2
    # The peak is the maximum of autoconv_lin
    
    # Normalize: the "normalized maximum" likely means max(autoconv) / (sum(f))^2
    # or max(autoconv) / integral(f)^2. Let's use the standard normalization.
    # If f is a step function with heights h_i on intervals of width 1/N,
    # then integral(f) = (1/N) * sum(h_i).
    # The autoconvolution (f*f)(t) has units of height^2 * width.
    # Normalized max = max(f*f) / (integral(f))^2
    
    integral_f = np.sum(f) / N  # approximate integral
    if integral_f < 1e-12:
        return 0.0
    
    max_autoconv = np.max(autoconv_lin)
    # The discrete autoconv_lin[k] approximates (f*f)(t_k) * (1/N)
    # because the convolution sum approximates the integral with step 1/N
    # So (f*f)(t) ≈ N * autoconv_lin[k]
    # And integral(f)^2 = (sum(f)/N)^2
    # Normalized max = (N * max_autoconv) / (sum(f)/N)^2
    #                = N^3 * max_autoconv / (sum(f))^2
    
    # Alternatively, if we think of f as a density, the normalized autoconv
    # is max(f*f) / (int f)^2. Let's just use the ratio that's scale-invariant.
    
    # Actually, let me reconsider. The task says "normalized maximum of its 
    # autoconvolution". A common normalization is max(f*f) / ||f||_1^2.
    
    # Let's compute it as: max of the continuous autoconvolution divided by 
    # (integral of f)^2.
    
    # The discrete autoconv_lin[k] = sum_j f[j] * f[k-j]
    # This approximates N * (f*f)(t_k) where t_k = (k - N + 1)/N * ... 
    # Hmm, let me just use the ratio max(autoconv_lin) / (sum(f)/N)^2 * (1/N)
    # No, let's keep it simple and scale-invariant.
    
    # The quantity we want to minimize is scale-invariant in f, so any 
    # consistent normalization works. Let's use:
    # score = max(autoconv_lin) / (sum(f))^2 * N^2
    # This is invariant to scaling of f.
    
    sum_f = np.sum(f)
    if sum_f < 1e-12:
        return 0.0
    
    score = max_autoconv / (sum_f * sum_f) * N * N
    return score

def _objective(coeffs):
    """Objective function for optimization. coeffs are raw (can be negative),
    but we clip to non-negative internally."""
    c = np.maximum(coeffs, 0.0)
    return _autoconv_max(c)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = _autoconv_max(np.ones(K))
    best_coeffs = np.ones(K)
    
    # Try multiple random initializations with Nelder-Mead
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        
        # Random non-negative initial coefficients
        np.random.seed(n_restarts * 7919)
        init = np.random.exponential(1.0, size=K)
        
        # Nelder-Mead optimization
        result = minimize(
            _objective,
            init,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            }
        )
        
        c_opt = np.maximum(result.x, 0.0)
        score = _autoconv_max(c_opt)
        
        if score < best_score:
            best_score = score
            best_coeffs = c_opt.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Reconstruct the step function from the best coefficients
    f = best_coeffs @ B
    f = np.maximum(f, 0.0)
    
    # Return as a list of 600 non-negative step heights
    return list(f)

# Pre-compute at import time to ensure it works
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Min: {min(result):.6f}, Max: {max(result):.6f}")
    print(f"Sum: {sum(result):.6f}")
