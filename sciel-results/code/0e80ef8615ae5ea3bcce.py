"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconv_fft(h):
    """Compute autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.fft(h, size=size)
    C = np.fft.ifft(H * np.conj(H)).real
    return C


def _objective(h, beta=100.0):
    """Soft-max of absolute sidelobes relative to peak."""
    c = _autoconv_fft(h)
    peak = c[N-1]
    if peak < 1e-15:
        return 1e10
    sidelobes = np.abs(c)
    sidelobes[N-1] = 0
    m = np.max(sidelobes)
    if m < 1e-15:
        return 0.0
    # Soft-max
    log_sum_exp = m + np.log(np.sum(np.exp((sidelobes - m) * beta))) / beta
    return log_sum_exp / peak


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Initialize with a window
    t = np.linspace(-1, 1, N)
    h0 = 1.0 - 0.5 * t**2  # parabolic taper
    h0 = h0 / np.max(np.abs(h0))
    
    best_h = h0.copy()
    best_obj = _objective(h0)
    
    # Try a few different initializations and optimize each
    inits = [
        h0,
        np.ones(N),
        np.cos(np.pi * t / 2),  # half-cosine
        np.exp(-2 * t**2),  # Gaussian-like
    ]
    
    for h_init in inits:
        if time.time() > deadline:
            break
        
        h_init = h_init / (np.max(np.abs(h_init)) + 1e-15)
        
        result = minimize(
            _objective,
            h_init,
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_h = result.x
    
    return best_h
# EVOLVE-BLOCK-END
