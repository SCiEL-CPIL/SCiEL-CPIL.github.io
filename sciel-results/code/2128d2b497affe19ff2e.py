import numpy as np
import time

N = 600
DELTA = 1.0 / (2.0 * N)

def _objective(h):
    """
    Calculate the normalized maximum autoconvolution.
    J = max(f*f) / (int f)^2
    """
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # Sum of heights
    s = np.sum(h)
    if s == 0:
        return 1e9
        
    # Autoconvolution using FFT
    # Length 2N is sufficient for linear convolution of two length N sequences
    n_fft = 2 * N
    F = np.fft.fft(h, n=n_fft)
    # Power spectrum
    power = F * np.conj(F)
    # Inverse FFT to get correlation/convolution
    # Note: np.fft.ifft of |F|^2 gives circular autocorrelation.
    # With zero padding to 2N, it gives linear autocorrelation.
    corr = np.fft.ifft(power).real
    
    # The maximum of the autoconvolution is at lag 0 (index 0) for non-negative h
    max_conv = corr[0]
    
    # Normalization
    # (int f)^2 = (DELTA * s)^2
    # max(f*f) = DELTA * max_conv
    # Ratio = (DELTA * max_conv) / (DELTA^2 * s^2) = max_conv / (DELTA * s^2)
    return max_conv / (DELTA * s * s)

def construct():
    start_time = time.time()
    
    # Candidate 1: Uniform
    h_uniform = np.ones(N)
    val_uniform = _objective(h_uniform)
    
    # Candidate 2: Gaussian
    # Choose sigma such that it fits well within N
    # Try a few sigmas
    best_h = h_uniform
    best_val = val_uniform
    
    x = np.arange(N)
    center = N / 2.0
    
    for sigma_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        sigma = N * sigma_frac
        h_gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        val_gauss = _objective(h_gauss)
        
        if val_gauss < best_val:
            best_val = val_gauss
            best_h = h_gauss
            
    # Time check
    if time.time() - start_time > 100:
        return list(best_h)
        
    return list(best_h)
