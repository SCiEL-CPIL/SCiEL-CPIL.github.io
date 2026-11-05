import numpy as np

N = 400

def construct():
    # Design h via spectral inversion to minimize peak-to-average ratio of autoconvolution
    # Target: minimize max(c) / (sum(h))^2 where c = convolve(h, h)
    
    # Step 1: Initialize with a smooth spectral shape using Wiener-Hopf inspired design
    # We want |H|^2 to be shaped such that its inverse FFT (autocorrelation) has minimal peak
    
    # Start with a raised-cosine-like spectrum that tapers to zero at band edges
    freqs = np.fft.fftfreq(N)
    
    # Design target power spectral density: smooth bell shape centered at DC
    # Use a Gaussian-like shape in frequency domain
    sigma_f = 0.15  # Control bandwidth
    target_PSD = np.exp(-2 * (freqs / sigma_f) ** 2)
    
    # Ensure it's real and non-negative
    target_PSD = np.maximum(target_PSD, 1e-10)
    
    # Initialize h from the target PSD: h = IFFT(sqrt(target_PSD))
    # But we need h to be real, so use the magnitude spectrum
    H_init = np.sqrt(target_PSD)
    h = np.real(np.fft.ifft(H_init))
    
    # Normalize
    h = h / np.sum(h)
    
    # Step 2: Refine via gradient descent on the actual objective
    # Objective: value = abs(2*N*max(c) / sum(h)^2)
    # We minimize max(c) while keeping sum(h) fixed (normalize after each step)
    
    best_h = h.copy()
    best_value = 2.0 * N * np.max(np.convolve(h, h)) / np.sum(h)**2
    
    # Gradient descent with adaptive step size
    eta = 0.01
    momentum = np.zeros(N)
    mu = 0.9
    
    for iteration in range(5000):
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        
        # Compute gradient of max(c) w.r.t. h
        # c_k = sum_j h_j h_{k-j}
        # d(c_max)/d(h_i) = h_{max_idx - i} + h_{max_idx - (N-1-i)} if in range
        # More precisely: dc_k/dh_i = h_{k-i} + h_{k-(N-1-i)} for valid indices
        
        grad = np.zeros(N)
        for i in range(N):
            # Contribution to c[max_idx] from h[i]
            j1 = max_idx - i
            j2 = max_idx - (N - 1 - i)
            
            val = 0.0
            if 0 <= j1 < N:
                val += h[j1]
            if 0 <= j2 < N and j2 != j1:
                val += h[j2]
            elif 0 <= j2 < N:
                # If j2 == j1, we already counted it once, but derivative should count both terms if they're the same variable
                # Actually dc_k/dh_i = sum over all pairs (j, k-j) where one is i
                # For c_k = sum_j h_j h_{k-j}, d/dh_i = h_{k-i} + h_{k-i} if k-i == k-i (always true)
                # Wait: c_k = sum_j h_j h_{k-j}
                # d/dh_i c_k = h_{k-i} + h_{k-i} = 2*h_{k-i} if i is in the sum
                # But the sum is over j from 0 to N-1, so h_i appears when j=i (giving h_{k-i}) and when k-j=i i.e. j=k-i (giving h_i)
                # So if i != k-i, we have two terms: h_{k-i} and h_i
                # If i == k-i, we have 2*h_i
                pass

        # Simpler: use the fact that d(convolve(h,h)[k])/dh[i] = h[k-i] + h[k-(N-1-i)] with boundary handling
        # Let me use a cleaner derivation:
        # c = convolve(h, h, 'full') has length 2N-1
        # c[k] = sum_{i} h[i] * h[k-i] where h is extended with zeros
        # d c[k] / d h[i] = h[k-i] + h[k-i] if we consider h as appearing in both factors
        # Actually: c[k] = sum_{j=0}^{N-1} h[j] * h[k-j] (with h out of range = 0)
        # d/dh[i] c[k] = h[k-i] + h[k-i] = 2*h[k-i] if 0<=k-i<N, else 0
        # Wait, that's not right either. Let me be careful:
        # c[k] = sum_j h[j] h[k-j]
        # d/dh[i] = sum_j (delta_{j,i} h[k-j] + h[j] delta_{k-j,i})
        #         = h[k-i] + h[k-i] if k-i is in [0,N-1]
        #         = 2*h[k-i] if 0 <= k-i < N
        
        grad = np.zeros(N)
        for i in range(N):
            k_minus_i = max_idx - i
            if 0 <= k_minus_i < N:
                grad[i] = 2.0 * h[k_minus_i]
        
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
