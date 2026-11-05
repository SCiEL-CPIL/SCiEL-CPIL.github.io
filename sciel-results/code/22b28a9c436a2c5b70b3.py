"""Optimized step heights for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_value(h):
    """Compute the evaluator's value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _objective(params, K, h0_scale):
    """
    Objective in Fourier-basis parameter space.
    params: real vector of length K representing Fourier coefficients.
    We use a cosine/sine basis to keep h real.
    """
    n = np.arange(N)
    # Build h from Fourier coefficients: h[k] = sum_j a_j * cos(2*pi*j*k/N) + b_j * sin(2*pi*j*k/N)
    # For simplicity, use only cosines (even symmetry) to reduce parameters
    # Actually, let's use both for more flexibility
    K_half = K // 2
    h = np.zeros(N)
    if K_half > 0:
        cos_terms = np.cos(2.0 * np.pi * np.outer(n, np.arange(K_half)) / N)
        h += params[:K_half] @ cos_terms
    if K - K_half > 0:
        sin_terms = np.sin(2.0 * np.pi * np.outer(n, np.arange(K - K_half)) / N)
        h += params[K_half:] @ sin_terms
    
    # Ensure sum(h) is not zero; add a small DC component if needed
    s = np.sum(h)
    if abs(s) < 1e-10:
        # Add a tiny DC offset
        h = h + 1e-6
    
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    s2 = np.sum(h) ** 2
    
    if s2 <= 0:
        return 1e10
    
    value = abs(2.0 * N * max_val / s2)
    return value


def _objective_smooth(params, K, beta):
    """Smooth objective using log-sum-exp for the max."""
    n = np.arange(N)
    K_half = K // 2
    h = np.zeros(N)
    if K_half > 0:
        cos_terms = np.cos(2.0 * np.pi * np.outer(n, np.arange(K_half)) / N)
        h += params[:K_half] @ cos_terms
    if K - K_half > 0:
        sin_terms = np.sin(2.0 * np.pi * np.outer(n, np.arange(K - K_half)) / N)
        h += params[K_half:] @ sin_terms
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 1e-6
        s = np.sum(h)
    
    # Use FFT to compute autocorrelation efficiently
    H = np.fft.fft(h, n=2*N)
    auto = np.fft.ifft(H * np.conj(H)).real
    # The full convolution has length 2N-1, but autocorrelation is symmetric
    # np.convolve(h,h) has length 2N-1, indices 0..2N-2
    # The autocorrelation (circular) would be different, but let's use linear
    # Actually, for the max, we need the full linear convolution
    # Let's just compute it directly for accuracy in the objective
    conv = np.convolve(h, h)
    
    # Log-sum-exp for smooth max
    max_conv = np.max(conv)
    if max_conv > 0:
        lse = max_conv + np.log(np.sum(np.exp((conv - max_conv) / beta))) * beta
    else:
        lse = max_conv  # if all negative, max is the max
    
    s2 = s * s
    if s2 <= 0:
        return 1e10
    
    return abs(2.0 * N * lse / s2)


def _objective_fft(params, K):
    """
    Faster objective using FFT for autocorrelation.
    The linear convolution h*h can be computed via FFT.
    """
    n = np.arange(N)
    K_half = K // 2
    h = np.zeros(N)
    if K_half > 0:
        cos_terms = np.cos(2.0 * np.pi * np.outer(n, np.arange(K_half)) / N)
        h += params[:K_half] @ cos_terms
    if K - K_half > 0:
        sin_terms = np.sin(2.0 * np.pi * np.outer(n, np.arange(K - K_half)) / N)
        h += params[K_half:] @ sin_terms
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 1e-6
        s = np.sum(h)
    
    # Linear convolution via FFT
    size = 2 * N - 1
    H = np.fft.fft(h, n=size)
    conv = np.fft.ifft(H * H).real
    
    max_val = np.max(conv)
    s2 = s * s
    if s2 <= 0:
        return 1e10
    
    return abs(2.0 * N * max_val / s2)


def _objective_fft_smooth(params, K, beta):
    """FFT-based objective with log-sum-exp smoothing."""
    n = np.arange(N)
    K_half = K // 2
    h = np.zeros(N)
    if K_half > 0:
        cos_terms = np.cos(2.0 * np.pi * np.outer(n, np.arange(K_half)) / N)
        h += params[:K_half] @ cos_terms
    if K - K_half > 0:
        sin_terms = np.sin(2.0 * np.pi * np.outer(n, np.arange(K - K_half)) / N)
        h += params[K_half:] @ sin_terms
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 1e-6
        s = np.sum(h)
    
    size = 2 * N - 1
    H = np.fft.fft(h, n=size)
    conv = np.fft.ifft(H * H).real
    
    # Log-sum-exp
    max_conv = np.max(conv)
    if max_conv > 0 and beta < 1e6:
        shifted = (conv - max_conv) / beta
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        lse = max_conv + beta * np.log(np.sum(np.exp(shifted)))
    else:
        lse = max_conv
    
    s2 = s * s
    if s2 <= 0:
        return 1e10
    
    return abs(2.0 * N * lse / s2)


def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Multi-start optimization in Fourier basis
    # Try different numbers of Fourier coefficients
    K_values = [8, 16, 32, 64]
    rng = np.random.default_rng(42)
    
    for K in K_values:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        n_starts = 10
        for i in range(n_starts):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Initialize parameters
            if i == 0:
                # Start from zero (DC only)
                params0 = np.zeros(K)
                params0[0] = 1.0  # DC component
            else:
                # Random initialization
                params0 = rng.normal(0, 0.1, size=K)
                # Ensure some DC component
                params0[0] += 1.0
            
            # Use smooth objective for optimization
            beta = 1.0  # smoothing parameter
            
            try:
                res = minimize(
                    _objective_fft_smooth,
                    params0,
                    args=(K, beta),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Evaluate the true (non-smooth) objective at the result
                h_opt = _params_to_h(res.x, K)
                val = _compute_value(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    
            except Exception:
                pass
    
    # Final polish: local optimization in h-space around the best solution
    if time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(
                lambda h: _compute_value(h),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if _compute_value(res.x) < best_val:
                best_h = res.x
                best_val = _compute_value(best_h)
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h


def _params_to_h(params, K):
    """Convert Fourier parameters to step heights."""
    n = np.arange(N)
    K_half = K // 2
    h = np.zeros(N)
    if K_half > 0:
        cos_terms = np.cos(2.0 * np.pi * np.outer(n, np.arange(K_half)) / N)
        h += params[:K_half] @ cos_terms
    if K - K_half > 0:
        sin_terms = np.sin(2.0 * np.pi * np.outer(n, np.arange(K - K_half)) / N)
        h += params[K_half:] @ sin_terms
    return h
