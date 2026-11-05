import numpy as np

N = 400

def construct():
    # Design h via spectral inversion to minimize peak-to-average ratio of autoconvolution
    # Target: minimize max(c) / (sum(h))^2 where c = convolve(h, h)
    
    # Step 1: Initialize with a smooth spectral shape using Wiener-Hopf inspired design
    # We want |H|^2 to be shaped such that its inverse FFT (autocorrelation) has minimal peak
    
    # Start with a raised-cosine-like spectrum that tapers to zero at band edges
    # Use a very narrow Gaussian-like shape in frequency domain to approximate a flat time-domain signal
    freqs = np.fft.fftfreq(N)
    
    # Design target power spectral density: smooth bell shape centered at DC
    # Use a Gaussian-like shape in frequency domain
    # Smaller sigma_f -> Narrower peak in freq -> Wider/Flatter peak in time
    sigma_f = 0.01  # Control bandwidth, very small to get close to constant h
    target_PSD = np.exp(-2 * (freqs / sigma_f) ** 2)
    
    # Ensure it's real and non-negative
    target_PSD = np.maximum(target_PSD, 1e-10)
    
    # Initialize h from the target PSD: h = IFFT(sqrt(target_PSD))
    # But we need h to be real, so use the magnitude spectrum
    H_init = np.sqrt(target_PSD)
    h = np.real(np.fft.ifft(H_init))
    
    # Normalize
    s = np.sum(h)
    if abs(s) < 1e-9:
        h[0] += 1e-9
        s = np.sum(h)
    h = h / s
    
    # Step 2: Refine via gradient descent on the actual objective
    # Objective: value = abs(2*N*max(c) / sum(h)^2)
    # We minimize max(c) while keeping sum(h) fixed (normalize after each step)
    
    best_h = h.copy()
    c = np.convolve(h, h)
    best_value = 2.0 * N * np.max(c) / np.sum(h)**2
    
    # Gradient descent with adaptive step size
    eta = 0.01
    momentum = np.zeros(N)
    mu = 0.9
    
    for iteration in range(500):
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        
        # Compute gradient of max(c) w.r.t. h
        # d c[k] / d h[i] = 2 * h[k-i] if 0 <= k-i < N, else 0
        grad = np.zeros(N)
        # Vectorized gradient calculation
        # i ranges from 0 to N-1
        # j = max_idx - i
        i_vals = np.arange(N)
        j_vals = max_idx - i_vals
        mask = (j_vals >= 0) & (j_vals < N)
        grad[i_vals[mask]] = 2.0 * h[j_vals[mask]]
        
        # Normalize gradient and apply momentum
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        momentum = mu * momentum - eta * grad
        h = h + momentum
        
        # Renormalize to keep sum(h) = 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
        
        # Track best
        c = np.convolve(h, h)
        current_value = 2.0 * N * np.max(c) / np.sum(h)**2
        if current_value < best_value:
            best_value = current_value
            best_h = h.copy()
    
    return best_h
