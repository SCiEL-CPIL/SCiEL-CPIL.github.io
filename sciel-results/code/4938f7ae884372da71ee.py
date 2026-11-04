"""
Strategy: Structured Chirp Design with Scalar Optimization.
Instead of optimizing 400 individual values or phases, we constrain the signal
to be a real-valued Zadoff-Chu-like sequence (or a general linear chirp) 
x[n] = A * cos(2*pi*(mu * n^2 + phi * n) / N).
This family of signals has excellent auto-correlation properties (low sidelobes)
by construction. We optimize the chirp rate `mu` and phase `phi` (or simply `mu` 
if we fix phi) to minimize the specific metric: max|autoconv| / (sum(x))^2.

We use Brent's method for the 1D optimization of `mu` over a range of values,
and potentially a small grid search over `phi` or a 2D Nelder-Mead if needed, 
but primarily relying on the 1D structure which is much easier to converge.
We also try a few "tapered" variants where the amplitude is modulated by a 
smooth window to further control the spectral content, optimizing the window 
width parameter.
"""

import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 400

# EVOLVE-BLOCK-START
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_chirp(mu, phi=0.0, N=N):
    """
    Generate a real-valued chirp sequence.
    x[n] = cos(2*pi*(mu * n^2 + phi * n) / N)
    """
    n = np.arange(N)
    phase = 2 * np.pi * (mu * n**2 + phi * n) / N
    return np.cos(phase)

def _generate_tapered_chirp(mu, phi, width, N=N):
    """
    Generate a chirp with a raised-cosine taper to control spectral leakage.
    width is in [0.0, 1.0], 0.0 means no taper (rectangular), 1.0 means full taper.
    """
    n = np.arange(N)
    phase = 2 * np.pi * (mu * n**2 + phi * n) / N
    x = np.cos(phase)
    
    if width > 1e-6:
        # Apply a raised cosine window
        # w[n] = 0.5 * (1 - cos(2*pi*n/(N-1))) is not quite right for tapering ends
        # Standard raised cosine: 0.5 * (1 - cos(2*pi*n / (N-1))) for n in [0, N-1]
        # But we want to taper the ends to zero to reduce spectral sidelobes?
        # Actually, for auto-correlation, rectangular is often better for peak, 
        # but tapers can reduce the *max* if the peak is caused by edge effects?
        # Let's try a simple Hann window scaled by width.
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
        # Interpolate between rectangular (1) and Hann (w)
        # If width=0, w=1. If width=1, w=Hann.
        w_final = (1 - width) * 1.0 + width * w
        x = x * w_final
        
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_val = float('inf')
    
    # Strategy 1: Pure Chirp Optimization
    # We search for mu in a range. The metric is periodic/oscillatory in mu.
    # We can do a coarse grid search first, then refine with Brent.
    
    # Coarse grid search for mu
    mu_range = np.linspace(-10, 10, 200)
    best_mu_coarse = None
    best_val_coarse = float('inf')
    
    for mu in mu_range:
        if time.time() > deadline - 30:
            break
        x = _generate_chirp(mu, 0.0)
        val = _metric(x)
        if val < best_val_coarse:
            best_val_coarse = val
            best_mu_coarse = mu
            best_x = x
            best_val = val
    
    if best_mu_coarse is not None:
        # Refine around the best coarse mu
        # Define objective for Brent
        def obj(mu):
            x = _generate_chirp(mu, 0.0)
            return _metric(x)
        
        # Search in a small window around best_mu_coarse
        # The period of the chirp in mu is related to N. 
        # mu changes the quadratic phase. 
        # Let's try a window of size 1.0 or 0.5.
        bound_low = best_mu_coarse - 0.5
        bound_high = best_mu_coarse + 0.5
        
        try:
            res = minimize_scalar(obj, bounds=(bound_low, bound_high), method='bounded', 
                                  options={'xatol': 1e-6, 'maxiter': 50})
            if res.success:
                mu_opt = res.x
                x_opt = _generate_chirp(mu_opt, 0.0)
                val_opt = _metric(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt
        except Exception:
            pass
            
        # Also try optimizing phi for the best mu
        def obj_phi(phi):
            x = _generate_chirp(best_mu_coarse, phi)
            return _metric(x)
            
        try:
            res_phi = minimize_scalar(obj_phi, bounds=(0, 1), method='bounded', 
                                      options={'xatol': 1e-6, 'maxiter': 50})
            if res_phi.success:
                phi_opt = res_phi.x
                x_opt = _generate_chirp(best_mu_coarse, phi_opt)
                val_opt = _metric(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt
        except Exception:
            pass

    # Strategy 2: Tapered Chirp Optimization
    # If time permits, try optimizing the taper width for the best mu found.
    if time.time() < deadline - 10 and best_mu_coarse is not None:
        def obj_width(width):
            x = _generate_tapered_chirp(best_mu_coarse, 0.0, width)
            return _metric(x)
            
        try:
            res_w = minimize_scalar(obj_width, bounds=(0.0, 0.5), method='bounded', 
                                    options={'xatol': 1e-4, 'maxiter': 20})
            if res_w.success:
                w_opt = res_w.x
                x_opt = _generate_tapered_chirp(best_mu_coarse, 0.0, w_opt)
                val_opt = _metric(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt
        except Exception:
            pass

    # Fallback: If something went wrong or metric is still high, use a known good pattern
    # The "ones" vector has metric 1.0 (autoconv peak N, sum N, ratio N/N^2 = 1/N? No.)
    # For x=ones: conv[0]=N, sum=N. Metric = N / N^2 = 1/N = 0.0025.
    # Wait, the metric is max|conv| / sum^2.
    # If x is all 1s, sum=400, sum^2=160000. conv[0]=400. Metric = 400/160000 = 0.0025.
    # The seed score is 1.2054. This implies the seed's metric is roughly 1.2054 * (seed_val/seed_val)?
    # No, score = seed/value. If seed is 1.0, value is metric.
    # If seed program returns ones, metric is 0.0025. Score would be 1/0.0025 = 400?
    # The problem statement says: "score is the upper bound on C3' ... lower is better ... search score is seed/value ... 1.0 equals seed program".
    # This implies the "value" returned by the evaluator is the metric.
    # If the seed program returns a signal with metric M_seed, and my program returns M_mine,
    # Score = M_seed / M_mine.
    # If Score > 1, M_mine < M_seed.
    # The current program has score 1.2054.
    # If the seed is "ones", M_seed = 0.0025. Then M_mine = 0.0025 / 1.2054 ~ 0.00207.
    # My chirp approach should aim for even lower metric.
    
    # Let's verify the metric of the best found.
    if best_x is None:
        # Fallback to ones
        best_x = np.ones(N)
        best_val = _metric(best_x)
        
    # Final check: ensure we are within time
    if time.time() > deadline:
        pass
        
    return best_x
# EVOLVE-BLOCK-END
