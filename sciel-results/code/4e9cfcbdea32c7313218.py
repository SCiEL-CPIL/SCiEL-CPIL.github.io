import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def compute_score(f):
    # f is array of heights
    # Convolution
    # Use FFT for speed
    n = len(f)
    # Pad to power of 2 for FFT
    size = 1
    while size < 2*n:
        size *= 2
    
    f_fft = np.fft.fft(f, size)
    conv_fft = f_fft * f_fft
    conv = np.fft.ifft(conv_fft).real[:2*n-1]
    
    max_conv = np.max(conv)
    l1_norm = np.sum(f)
    l1_sq = l1_norm * l1_norm
    
    if l1_sq == 0:
        return float('inf')
        
    return max_conv / l1_sq

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback to seed
        return [1.0] * N
        
    # Try different Gaussian widths
    # x is from -0.25 to 0.25
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N # Centers
    
    best_f = [1.0] * N
    best_score = compute_score(best_f) # Seed score ~ 2.0
    
    # Sample sigmas
    # Support is 0.5 wide.
    # Sigma should be around 0.1 - 0.2
    sigmas = np.linspace(0.05, 0.25, 20)
    
    for sigma in sigmas:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Gaussian
        f = np.exp(-x**2 / (2 * sigma**2))
        
        # Ensure non-negative (it is)
        # Calculate score
        score = compute_score(f)
        
        if score < best_score:
            best_score = score
            best_f = f
            
    # Refine around best sigma if time permits
    # ... (simple refinement)
    
    # Return list of floats
    return list(best_f)
