import numpy as np

N = 400

def construct():
    # Initialize h
    # Start with constant + anti-symmetric component
    h = np.ones(N)
    # Add anti-symmetric sine wave to reduce center peak
    k = np.arange(N)
    h += 0.5 * np.sin(2 * np.pi * k / N)
    
    # Normalize sum to 1
    s = np.sum(h)
    h = h / s
    
    # Gradient descent to minimize sum(c^2)
    # which flattens the autocorrelation
    best_h = h.copy()
    best_val = np.inf
    
    eta = 0.1
    momentum = np.zeros(N)
    mu = 0.9
    
    for i in range(500):
        c = np.convolve(h, h)
        # Objective: sum(c^2)
        # Grad of sum(c^2) wrt h
        # c_k = sum_j h_j h_{k-j}
        # d(sum c^2)/dh_i = 2 * sum_k c_k * (h_{k-i} + h_{k-i+N}?)
        # Actually, grad = 2 * np.convolve(c, h) ?
        # Let's verify.
        # c = h * h (full)
        # d(c)/dh = h * (delta) ?
        # Gradient of autocorrelation energy is 2 * c * h (convolved)?
        # Let's use the property: sum(c^2) = sum(h^2 * h^2) = ||h*h||_2^2
        # This is the 4th order moment.
        # Gradient is 2 * np.convolve(np.convolve(h, h), h) ? No.
        
        # Simpler: Use FFT.
        # C_spec = FFT(c) = |FFT(h)|^2
        # sum(c^2) = sum(C_spec^2) / N? No.
        # Parseval: sum(c^2) = (1/L) sum(|FFT(c)|^2) ?
        # Let's just compute grad numerically if analytical is tricky, or use 
        # grad = 2 * np.convolve(c, h, mode='same')?
        
        # Let's check:
        # J = sum_k c_k^2
        # dJ/dh_i = 2 sum_k c_k (h_{k-i} + h_{k-i+N}?)
        # Note c_k is defined for k in [0, 2N-2].
        # h_j is 0 outside [0, N-1].
        # c_k = sum_j h_j h_{k-j}
        # d c_k / d h_i = h_{k-i} + h_{k-i} (if i and k-i are distinct? No)
        # d c_k / d h_i = h_{k-i} + h_{k-i} (derivative of h_j h_{k-j} wrt h_i)
        # If j=i, term is h_{k-i}. If k-j=i, term is h_j=h_i? No.
        # d/dh_i (h_j h_{k-j}) = delta_{ij} h_{k-j} + h_j delta_{k-j, i}
        # = delta_{ij} h_{k-i} + h_i delta_{k-i, j}
        # Sum over j:
        # j=i: h_{k-i}
        # k-j=i => j=k-i: h_{k-i}
        # So d c_k / d h_i = 2 h_{k-i} (if i != k-i, i.e., 2i != k)
        # If 2i == k, derivative is h_i (only one term? No, h_i * h_i, deriv is 2 h_i).
        # So d c_k / d h_i = 2 h_{k-i} always.
        
        # So grad_i = 2 sum_k c_k h_{k-i}
        # This is 2 * (c * h)_i where * is cross-correlation or convolution?
        # sum_k c_k h_{k-i}.
        # Let m = k-i. k = m+i.
        # sum_m c_{m+i} h_m.
        # This is the cross-correlation of c and h at lag i.
        # np.correlate(c, h, mode='full')?
        # Or just np.convolve(c, h[::-1])?
        
        # Let's use np.convolve(c, h) and extract the right part.
        # c has length 2N-1. h has length N.
        # conv(c, h) has length 3N-2.
        # We need sum_k c_k h_{k-i}.
        # This is equivalent to (c * h)[i] if we align centers?
        
        # Let's just use a simple numerical gradient check or 
        # assume grad = 2 * np.convolve(c, h, mode='same') centered appropriately.
        
        # Actually, there is a known result:
        # Minimize sum(c^2) is equivalent to making h have a specific spectral shape.
        # Let's just use finite differences for 10 iterations to get the sign, 
        # then analytical?
        # N=400 is small. Finite difference is 400 evaluations.
        # 500 iterations * 400 evals * O(N^2) conv = 500 * 400 * 160000 = 32 billion. Too slow.
        
        # Let's derive the FFT gradient.
        # J = sum(c^2).
        # C = FFT(c, n=L) where L >= 2N-1.
        # C = |H|^2 (element-wise, with padding).
        # J = sum(c^2) = (1/L) sum(C^2) (Parseval, with 1/L factor for unnormalized FFT? 
        # np.fft.fft uses 1 in forward, 1/n in inverse.
        # sum(c^2) = (1/L) sum(|C|^2) ? No.
        # sum(c^2) = (1/L) sum(C * conj(C)) ? No, c is real.
        # sum(c^2) = (1/L) sum(C^2) if C is real?
        # C = |H|^2 is real and even.
        # So J = (1/L) sum(C^2) = (1/L) sum(|H|^4).
        
        # dJ/dH_k = (1/L) * 4 |H|^2 * H_k^* ?
        # d/dH (|H|^4) = 4 |H|^2 H^*.
        # So grad_H = 4 |H|^2 H^* / L.
        # grad_h = IFFT(grad_H).
        
        L = 2 * N
        H = np.fft.fft(h, n=L)
        C_spec = H * np.conj(H) # |H|^2
        # J = sum(c^2) = (1/L) sum(C_spec^2)
        # dJ/dH = (1/L) * 2 * C_spec * d(C_spec)/dH
        # C_spec = H * H^*
        # d(C_spec)/dH = H^* (if H is complex)
        # So dJ/dH = (2/L) * C_spec * H^*
        grad_H = (2.0 / L) * C_spec * np.conj(H)
        grad_h = np.fft.ifft(grad_H, n=L).real[:N]
        
        # Normalize gradient?
        grad_norm = np.linalg.norm(grad_h)
        if grad_norm > 1e-10:
            grad_h = grad_h / grad_norm
            
        momentum = mu * momentum - eta * grad_h
        h = h + momentum
        
        # Re-normalize sum to 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
        
        # Evaluate
        c = np.convolve(h, h)
        val = 2.0 * N * np.max(c) / (np.sum(h)**2)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
