import numpy as np
import time

N = 600
# Interval width
dx = 0.5 / N
# Midpoints
x = -0.25 + (np.arange(N) + 0.5) * dx

def compute_score(h):
    # h is a numpy array of heights
    # Autoconvolution
    # Using FFT for speed, or direct if small
    # N=600, direct is fine, but FFT is O(N log N)
    # Let's use direct convolution via np.convolve or fft
    # np.convolve(h, h, mode='full') gives length 2N-1
    # The convolution corresponds to the integral scaled by dx^2?
    # Integral approx: sum h_i h_j dx^2
    # But the "normalized maximum" usually implies the continuous function value.
    # If h_i is the height, the area is sum h_i dx.
    # The autoconvolution value at a point is approx sum h_i h_{i+k} dx.
    # Wait, the discrete convolution c_k = sum h_i h_{i+k}.
    # The continuous convolution value is c_k * dx.
    # L1 norm is sum h_i dx.
    # L1^2 is (sum h_i dx)^2.
    # Score = max(c_k * dx) / (sum h_i dx)^2
    #       = max(c_k) / (sum h_i)^2 * (1/dx) ? No.
    # max(c_k * dx) / (S * dx)^2 = max(c_k) * dx / (S^2 dx^2) = max(c_k) / (S^2 dx)
    
    c = np.convolve(h, h, mode='full')
    max_c = np.max(c)
    sum_h = np.sum(h)
    # Score = max_c / (sum_h * sum_h * dx)
    score = max_c / (sum_h * sum_h * dx)
    return score

def construct():
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    best_score = float('inf')
    best_h = np.ones(N)
    
    # Family 1: Gaussian
    # h_i = exp(-x^2 / (2 sigma^2))
    sigmas = np.logspace(-3, 0, 50) # from 0.001 to 1.0
    
    for sigma in sigmas:
        if time.time() > deadline:
            break
        h = np.exp(-x**2 / (2 * sigma**2))
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Family 2: Polynomial bump (1 - (x/w)^2)^k
    widths = np.linspace(0.05, 0.25, 20)
    powers = np.linspace(1, 5, 10)
    
    for w in widths:
        if time.time() > deadline:
            break
        for k in powers:
            h = np.zeros(N)
            mask = np.abs(x) < w
            h[mask] = (1 - (x[mask]/w)**2)**k
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                
    # Return as list of floats
    return [float(v) for v in best_h]
