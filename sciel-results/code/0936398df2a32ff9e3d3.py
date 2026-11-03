import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 140.0  # safety margin before 150s kill

def _autoconv_peak(h):
    """Compute the normalized maximum of the autoconvolution of h.
    
    h is an array of length N (step heights on equal intervals of [-1/4, 1/4]).
    The interval width is (1/4 - (-1/4)) / N = 1/2 / N = 1/(2N).
    
    The autoconvolution (convolution of f with itself) at shift k is:
        (f * f)(k) = sum_i f[i] * f[i-k]  (with appropriate scaling)
    
    For a step function on intervals of width dx, the continuous autoconvolution is:
        (f*f)(t) = integral f(s) f(t-s) ds
    
    If f is constant on each interval j with value h[j], then:
        (f*f)(t) = sum_{j,k: interval overlap} h[j]*h[k] * (overlap_length)
    
    For discrete approximation with dx = 1/(2N):
        (f*f)[m] ≈ dx * sum_i h[i] * h[i-m]  for m = -(N-1), ..., N-1
    
    The "normalized maximum" likely refers to max_t (f*f)(t) / (max_t (f*f)(t)) or similar.
    
    Actually, looking at the task: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    
    The constant C1 in the context of autocorrelation inequalities typically relates to:
        max |autocorrelation| / (integral |f|^2) or similar normalized quantity.
    
    Let me think more carefully. The autoconvolution of f with itself is (f*f)(t) = ∫ f(s)f(t-s)ds.
    
    For a nonnegative step function, the "normalized maximum" is likely:
        max_t (f*f)(t) / (∫ f(s) ds)^2
    
    This is because ∫ f(s) ds is the "mass" of the function, and normalizing by its square makes the quantity scale-invariant.
    
    Let dx = 1/(2N). Then:
        mass = dx * sum(h)
        autoconv[m] = dx * sum_i h[i] * h[i+m]  (for appropriate m)
        normalized_max = max_m autoconv[m] / mass^2
    """
    dx = 1.0 / (2.0 * N)
    mass = dx * np.sum(h)
    if mass <= 0:
        return 1e10
    # Compute autoconvolution via FFT for efficiency
    # Pad to avoid circular convolution
    size = 2 * N
    fft_h = np.fft.rfft(h, n=size)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=size)
    # The linear autoconvolution is in autoconv[0:N] and autoconv[size-N+1:size]
    # Actually, for real signals, the autoconvolution via IFFT of |FFT|^2 gives circular conv.
    # We padded to size=2N, so the linear autoconvolution is in autoconv[0:N] (positive shifts)
    # and we need to also consider negative shifts, but by symmetry autoconv[-m] = autoconv[m].
    # The peak is at m=0 typically for nonnegative functions, but let's check all.
    peak = np.max(autoconv[:N])  # This includes m=0 to m=N-1
    # Actually, the full linear autoconvolution has 2N-1 points. With padding to 2N:
    # autoconv[0] = sum h[i]^2 (m=0)
    # autoconv[1] = sum h[i]h[i+1] (m=1)
    # ...
    # autoconv[N-1] = sum h[i]h[i+N-1]
    # autoconv[N] = 0 (or small due to numerical error)
    # The negative shifts are symmetric, so max is in [0, N-1].
    return peak * dx / (mass * mass)

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _autoconv_peak(h0)
    
    # Use scipy minimize with BFGS or L-BFGS-B
    # Constraints: h[i] >= 0
    bounds = [(0.0, None)] * N
    
    def objective(h):
        return _autoconv_peak(h)
    
    # Try a few different initializations and optimization strategies
    initializations = [
        np.ones(N),
        np.ones(N) * (1 + 0.1 * np.random.randn(N)),
        np.exp(-10 * (np.linspace(-1, 1, N))**2),  # Gaussian-like
    ]
    
    for h_init in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        h_init = np.clip(h_init, 0, None)
        
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Polish pass with tighter tolerance
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.success or result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    return list(best_h)
