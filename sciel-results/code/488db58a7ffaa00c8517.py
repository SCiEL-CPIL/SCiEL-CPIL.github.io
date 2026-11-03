import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval [-1/4, 1/4]
x = np.linspace(-0.25, 0.25, N)
dx = x[1] - x[0]

def calculate_ratio(h):
    # h is the step heights
    # Ensure non-negative
    h = np.maximum(h, 0)
    norm2 = np.sum(h**2) * dx
    if norm2 == 0:
        return np.inf
    # Compute autoconvolution
    # Using FFT for speed
    # Pad to avoid circular convolution effects
    size = 2 * N
    h_fft = np.fft.fft(h, n=size)
    conv_fft = h_fft * h_fft
    conv = np.fft.ifft(conv_fft).real
    conv = conv[:N] # Keep the relevant part? 
    # Actually, linear convolution length is 2N-1.
    # We need the max of the full linear convolution.
    conv_full = np.fft.ifft(conv_fft).real
    max_conv = np.max(conv_full)
    # The integral of h*h is approximated by sum * dx
    # Wait, the convolution of two step functions:
    # (h*h)(t) = sum h_i h_j dx where t = x_i + x_j?
    # Discrete convolution C_k = sum h_i h_{k-i}.
    # The continuous value is approx C_k * dx.
    # So max continuous is max(C) * dx.
    # Norm2 is sum(h^2) * dx.
    # Ratio = (max(C) * dx) / (sum(h^2) * dx) = max(C) / sum(h^2).
    return max_conv / (norm2 / dx) # This is max(C)/sum(h^2)

def fourier_to_h(params):
    # params: [c0, c1, p1, c2, p2, ...]
    c0 = params[0]
    h = np.full(N, c0)
    for i in range(1, len(params)//2 + 1):
        c = params[2*i - 1]
        p = params[2*i]
        freq = i * 2 * np.pi / 0.5 # Period 0.5
        h += c * np.cos(freq * x + p)
    # Ensure non-negative by clipping or shifting?
    # Clipping changes the shape. 
    # Better to use a parameterization that guarantees non-negativity.
    # h = c0 + ...
    # If we just clip, the gradient is zero in clipped regions.
    # Let's return h, and let the optimizer handle it?
    # No, we must return valid h.
    return np.maximum(h, 0)

def objective(params):
    h = fourier_to_h(params)
    return calculate_ratio(h)

def construct():
    deadline = time.time() + 100.0
    best_h = np.ones(N)
    best_score = calculate_ratio(best_h)
    
    # Initial guess: Uniform
    # Try a few random Fourier initializations
    np.random.seed(42)
    
    n_modes = 5
    dim = 1 + 2 * n_modes
    
    for restart in range(20):
        if time.time() > deadline:
            break
        # Random init
        c0 = 1.0
        params = np.zeros(dim)
        params[0] = c0
        for i in range(1, n_modes + 1):
            params[2*i - 1] = np.random.uniform(-0.5, 0.5)
            params[2*i] = np.random.uniform(0, 2*np.pi)
            
        # Optimize
        res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_score:
            best_score = res.fun
            best_h = fourier_to_h(res.x)
            
    return list(best_h)
