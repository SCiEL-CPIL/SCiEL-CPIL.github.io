import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize with uniform
    f = np.ones(N) / N
    
    # Phase 1: Sinkhorn-Knopp spectral equalization
    # We want to minimize max(conv(f,f))
    # In Fourier space: F_hat(k) = FFT(f), and conv(f,f) has FFT = |F_hat(k)|^2
    # So we want |F_hat(k)|^2 to be as flat as possible
    # This is equivalent to making |F_hat(k)| constant, which means f is a delta
    # But we need f >= 0 and sum(f) = 1
    # 
    # Sinkhorn approach: iteratively rescale in Fourier domain to flatten spectrum
    # while maintaining the constraint that f is real and non-negative
    
    for iter in range(200):
        if time.time() > deadline:
            break
        
        # Compute FFT
        F = np.fft.rfft(f)
        
        # We want |F|^2 to be flat, so we want |F| to be constant
        # Target: |F(k)| = C for all k
        # But we need to maintain sum(f) = 1, which means F(0) = 1 (DC component)
        
        # Get current magnitudes
        mag = np.abs(F)
        target_mag = np.ones_like(mag)
        
        # Scale to make magnitudes more uniform
        # Use multiplicative update: F_new = F * (target / mag)^(1/2)
        # But this may make f complex or negative
        # 
        # Alternative: work in log-domain
        # Let F = |F| * exp(i*phase)
        # We want to adjust |F| to be flatter
        # 
        # Simple approach: compute a scaling factor that reduces peaks in |F|
        # and increases valleys
        
        # Compute target magnitude (geometric mean to preserve energy roughly)
        log_mag = np.log(mag + 1e-15)
        mean_log_mag = np.mean(log_mag)
        target_log_mag = mean_log_mag + 0.5 * (log_mag - mean_log_mag)
        # This blends current with flat, reducing contrast
        
        target_mag = np.exp(target_log_mag)
        
        # Apply scaling
        scale = target_mag / (mag + 1e-15)
        F_new = F * scale
        
        # Convert back to real
        f_new = np.fft.irfft(F_new, n=N)
        
        # Enforce non-negativity and normalization
        f_new = np.maximum(f_new, 0)
        f_new = f_new / (np.sum(f_new) + 1e-15)
        
        f = f_new
    
    # Phase 2: Gradient-based refinement to directly minimize peak autoconvolution
    for iter in range(500):
        if time.time() > deadline:
            break
        
        c = np.convolve(f, f)
        max_val = np.max(c)
        max_idx = np.argmax(c)
        
        # Subgradient: reduce f where it contributes most to the peak
        contrib = np.zeros(N)
        for i in range(N):
            j = max_idx - i
            if 0 <= j < N:
                contrib[i] = f[j]
        
        if np.sum(contrib) > 0:
            contrib_norm = contrib / np.sum(contrib)
        else:
            contrib_norm = np.ones(N) / N
        
        # Multiplicative update
        alpha = 0.05
        f = f * (1 - alpha * contrib_norm)
        f = np.clip(f, 0, None)
        f = f / (np.sum(f) + 1e-15)
    
    return list(f)
