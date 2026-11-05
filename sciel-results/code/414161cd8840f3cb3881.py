import numpy as np

N = 400

def construct():
    # Step 1: Design spectral shape via Wiener-Hopf to minimize peak autocorrelation
    # The key insight: we want |H(f)|^2 to be flat, but we can shape it to reduce
    # the peak of IFFT(|H|^2)
    
    L = 2 * N
    f = np.fft.fftfreq(L)
    
    # Initialize with a flat spectrum
    H = np.ones(L, dtype=complex)
    
    # Iteratively refine the spectral shape to minimize the peak
    for outer_iter in range(50):
        # Compute current autocorrelation
        C_spec = H * np.conj(H)
        c = np.fft.ifft(C_spec, n=L).real[:2*N-1]
        
        # Find the peak
        peak_val = np.max(c)
        if peak_val <= 0:
            break
        
        # Compute the gradient of the peak wrt C_spec
        # The peak is at some lag k, and c_k = sum_f C_spec(f) * exp(2*pi*i*f*k/L)
        # To reduce the peak, we want to reduce C_spec where it contributes most
        
        # Use iterative reweighting: C_spec_new = C_spec / (1 + alpha * c)
        # This reduces the spectrum where the autocorrelation is high
        
        alpha = 0.1
        C_spec_new = C_spec / (1.0 + alpha * c)
        
        # Ensure C_spec is non-negative
        C_spec_new = np.maximum(C_spec_new, 1e-10)
        
        # Update H
        H = np.sqrt(C_spec_new) * (H / np.abs(H + 1e-10))
        
        # Convert to time domain
        h = np.fft.ifft(H, n=L).real[:N]
        
        # Normalize sum to 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
    
    # Step 2: Refine with gradient descent
    best_h = h.copy()
    best_val = np.inf
    
    eta = 0.005
    momentum = np.zeros(N)
    mu = 0.9
    
    for i in range(100):
        c = np.convolve(h, h)
        
        # Compute gradient using soft max
        tau = 0.02
        c_shifted = c - np.max(c)
        weights = np.exp(c_shifted / tau)
        weights = weights / np.sum(weights)
        
        # grad_i = 2 * sum_k weights_k * h_{k-i}
        grad = 2.0 * np.correlate(weights, h, mode='same')[:N]
        
        s = np.sum(h)
        grad = grad / (s * s)
        
        momentum = mu * momentum - eta * grad
        h = h + momentum
        
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
        
        c = np.convolve(h, h)
        val = 2.0 * N * np.max(c) / (np.sum(h)**2)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        if i % 10 == 0:
            eta *= 0.9
    
    return best_h
