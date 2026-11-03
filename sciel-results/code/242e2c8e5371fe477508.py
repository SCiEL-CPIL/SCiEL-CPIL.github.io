import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize f
    # Start with a slightly perturbed uniform distribution to break symmetry
    f = np.ones(N) / N
    # Add small noise
    f += 0.01 * np.random.randn(N)
    f = np.abs(f)
    f /= np.linalg.norm(f)
    
    best_f = f.copy()
    best_val = np.inf
    
    # Parameters
    eta = 0.01  # Learning rate
    beta = 10.0  # Smoothing parameter for max
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() - start_time > deadline:
            break
            
        # Compute autoconvolution c
        F = np.fft.fft(f)
        C = np.fft.ifft(F * F)
        c = np.real(C)
        
        # Compute objective value (normalized max)
        # Assuming normalization is ||f||_2 = 1, so objective is max(c)
        current_val = np.max(c)
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            
        # Compute weights w for smoothed max
        # w_k = exp(beta * c_k) / sum(exp(beta * c_k))
        # To avoid overflow, shift by max
        c_shifted = c - np.max(c)
        w = np.exp(beta * c_shifted)
        w /= np.sum(w)
        
        # Compute gradient
        # grad_f = IFFT( (2/N) * F * FFT(w) )
        W = np.fft.fft(w)
        grad_F = (2.0 / N) * F * W
        grad_f = np.real(np.fft.ifft(grad_F))
        
        # Gradient descent step
        f = f - eta * grad_f
        
        # Enforce non-negativity
        f = np.maximum(f, 0)
        
        # Renormalize to keep ||f||_2 = 1
        norm = np.linalg.norm(f)
        if norm > 0:
            f /= norm
        else:
            f = np.ones(N) / N
            f /= np.linalg.norm(f)
            
        # Adaptive learning rate?
        # If value increases, reduce eta
        if current_val > best_val * 1.1:
            eta *= 0.9
        else:
            eta = min(eta * 1.01, 0.1)

    return list(best_f)
