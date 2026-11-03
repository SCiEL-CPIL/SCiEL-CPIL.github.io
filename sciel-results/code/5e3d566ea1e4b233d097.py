import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    # The autoconvolution at discrete points
    # (f*f)(t) for step function can be computed via convolution
    # Using FFT for efficiency
    h = np.asarray(h, dtype=np.float64)
    # Pad for linear convolution
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    # Scale: each unit of height contributes DELTA width
    # (f*f)(t) = DELTA^2 * sum over overlapping intervals
    # Actually, for step functions, the autoconvolution at grid points:
    # If f is constant h_i on interval i, then (f*f)(t) involves products h_i*h_j * overlap_length
    # The convolution of the discrete heights gives us the structure
    # conv[k] = sum_i h_i * h_{k-i} (discrete convolution)
    # The actual autoconvolution value at the k-th grid point of the output is:
    # DELTA * conv[k] * DELTA = DELTA^2 * conv[k]? No...
    # Let me think: f(x) = h_i for x in [x_i, x_{i+1}]
    # (f*f)(t) = ∫ f(s) f(t-s) ds
    # For step functions, this is piecewise linear (trapezoidal)
    # The peak values occur at the grid points
    # At t = k*DELTA, the autoconvolution is DELTA * sum_i h_i * h_{k-i} (approximately)
    # More precisely, if we sample at the right points...
    # For now, let's use: autoconv[k] = DELTA * conv[k] for k in range(size)
    # This is an approximation; the exact value at grid points for step functions
    # Actually, for a step function with constant values on intervals, the autoconvolution
    # at the midpoints or endpoints can be computed exactly.
    # Let's use the discrete convolution scaled by DELTA:
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    # score = max(f*f) * |I| / (∫f)^2
    # |I| = 1/2
    score = max_ac * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with uniform
    h = np.ones(N)
    best_score = compute_score(h)
    best_h = h.copy()
    
    # Try various parametric families
    # 1. Cosine taper: h_i = 1 + a*cos(pi*x_i)
    # 2. Gaussian-like
    # 3. Optimized via gradient descent
    
    # Let's try scipy minimize on a reduced parameterization
    # Use a low-dimensional parameterization: h_i = base * (1 + sum_j c_j * phi_j(x_i))
    
    # First, try simple parametric forms
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N  # midpoints
    
    # Try cosine shapes
    for a in np.linspace(-0.5, 0.5, 11):
        h_try = 1.0 + a * np.cos(2 * np.pi * x)
        h_try = np.maximum(h_try, 0)
        s = compute_score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()
    
    # Try Gaussian-like
    for sigma in np.linspace(0.05, 0.5, 11):
        h_try = np.exp(-x**2 / (2 * sigma**2))
        s = compute_score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()
    
    # Try trapezoidal shapes
    for w in np.linspace(0.1, 0.4, 11):
        h_try = np.ones(N)
        mask = np.abs(x) > (0.25 - w)
        h_try[mask] = 0
        s = compute_score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()
    
    # Now try gradient-based optimization
    if time.time() < deadline - 10:
        # Use a parameterization with a few Fourier modes
        def param_to_h(params):
            h = 1.0 + params[0] * np.cos(2*np.pi*x) + params[1] * np.sin(2*np.pi*x) + \
                params[2] * np.cos(4*np.pi*x) + params[3] * np.sin(4*np.pi*x) + \
                params[4] * np.cos(6*np.pi*x)
            return np.maximum(h, 0)
        
        def objective(params):
            h = param_to_h(params)
            return compute_score(h)
        
        x0 = np.zeros(5)
        result = optimize.minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            best_h = param_to_h(result.x).copy()
        
        # Try from different starting points
        for start in [np.random.randn(5)*0.1 for _ in range(5)]:
            if time.time() > deadline - 10:
                break
            result = optimize.minimize(objective, start, method='Nelder-Mead',
                                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_h = param_to_h(result.x).copy()
    
    # Final check
    if time.time() > deadline - 5:
        return best_h.tolist()
    
    return best_h.tolist()
