import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def objective(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        s = h.sum()
        if s == 0:
            return 1e10
        # Compute autoconvolution coefficients
        # For a discrete signal, autoconvolution at lag k is sum_i h[i]*h[i+k]
        # We need to compute this for all lags
        # Use FFT for efficiency
        # Pad to avoid circular convolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        fft_h = np.fft.fft(h_padded)
        autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        # autocorr[k] = sum_i h[i]*h[i+k] for k=0..N-1
        # Normalize by s^2
        norm_autocorr = autocorr / (s * s)
        return norm_autocorr.max()
    
    def objective_grad(h):
        # Numerical gradient
        h = np.maximum(h, 0)
        s = h.sum()
        if s == 0:
            return np.zeros(N)
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        fft_h = np.fft.fft(h_padded)
        autocorr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        norm_autocorr = autocorr / (s * s)
        max_idx = np.argmax(norm_autocorr)
        # Gradient of autocorr[k] w.r.t. h[i]
        # autocorr[k] = sum_j h[j]*h[j+k]
        # d(autocorr[k])/d(h[i]) = h[i+k] + h[i-k] (with appropriate bounds)
        grad_autocorr = np.zeros(N)
        k = max_idx
        for i in range(N):
            # Contribution from h[i]*h[i+k]
            if i + k < N:
                grad_autocorr[i] += h[i+k]
            # Contribution from h[i-k]*h[i]
            if i - k >= 0:
                grad_autocorr[i] += h[i-k]
        # d(norm_autocorr[k])/d(h[i]) = (d(autocorr[k])/d(h[i]) * s^2 - autocorr[k] * 2*s * h[i]) / s^4
        # = (grad_autocorr[i] * s - 2 * autocorr[k] * h[i]) / s^3
        grad = (grad_autocorr * s - 2 * autocorr[k] * h) / (s * s * s)
        return grad
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Try to minimize
    best_h = h0.copy()
    best_val = objective(h0)
    
    try:
        res = minimize(objective, h0, method='L-BFGS-B', 
                      bounds=[(0, None)] * N,
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < best_val:
            best_h = np.maximum(res.x, 0)
            best_val = objective(best_h)
    except:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return list(best_h)
    
    # Try a few different initializations
    inits = [
        np.ones(N),
        np.sin(np.pi * np.arange(N) / N) ** 2,  # bell-shaped
        np.exp(-((np.arange(N) - N/2) / (N/4)) ** 2),  # Gaussian
    ]
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        h_init = np.maximum(h_init, 0)
        if h_init.sum() == 0:
            continue
        try:
            res = minimize(objective, h_init, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_h = np.maximum(res.x, 0)
                best_val = objective(best_h)
        except:
            pass
    
    return list(best_h)
