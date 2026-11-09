import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 140.0  # seconds, safety margin before 150s limit

def construct():
    start_time = time.time()
    
    # Interval width
    h = 1.0 / (2 * N)  # each interval has width 1/1200
    
    # Fine grid for computing convolution
    # The autoconvolution of a function on [-1/4, 1/4] has support on [-1/2, 1/2]
    # Use a grid that's a power of 2 for FFT efficiency
    M = 4096  # fine grid size
    # Grid on [-1/2, 1/2]
    x_fine = np.linspace(-0.5, 0.5, M, endpoint=False)
    dx_fine = 1.0 / M
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    def objective(h_vec):
        # Ensure non-negative
        h_pos = np.maximum(h_vec, 0)
        
        # Upsample to fine grid
        # Each interval i covers [x_i, x_{i+1}) where x_i = -1/4 + i*h
        # Map fine grid points to interval indices
        # x_fine[k] is in interval i if -1/4 + i*h <= x_fine[k] < -1/4 + (i+1)*h
        # i = floor((x_fine[k] + 1/4) / h)
        idx = np.floor((x_fine + 0.25) / h).astype(int)
        # Clip to valid range
        idx = np.clip(idx, 0, N - 1)
        
        # f on fine grid
        f_fine = h_pos[idx]
        
        # Compute autoconvolution using FFT
        # f*f(t) = integral f(s) f(t-s) ds
        # Using FFT: FFT(f*f) = FFT(f)^2
        F = np.fft.fft(f_fine)
        conv = np.fft.ifft(F * F).real
        # Scale by dx_fine since FFT convolution gives sum, but we want integral
        conv *= dx_fine
        
        # The autoconvolution should be on [-1/2, 1/2]
        # Find max
        max_conv = np.max(conv)
        
        # Integral of f
        integral = np.sum(h_pos) * h
        
        if integral <= 0:
            return 1e10
        
        return max_conv / (integral * integral)
    
    # Use L-BFGS-B with bounds (non-negative)
    bounds = [(0, None)] * N
    
    # Try a few initializations
    best_h = h0.copy()
    best_obj = objective(h0)
    
    # Run optimization
    try:
        result = minimize(objective, h0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success or result.fun < best_obj:
            best_h = np.maximum(result.x, 0)
            best_obj = result.fun
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return list(best_h)
        
        # Try another initialization: Gaussian-like bump
        x = np.linspace(-0.25, 0.25, N)
        h_gauss = np.exp(-10 * x**2)
        h_gauss /= np.sum(h_gauss) * h  # normalize so integral = 1
        
        result2 = minimize(objective, h_gauss, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result2.fun < best_obj:
            best_h = np.maximum(result2.x, 0)
            best_obj = result2.fun
        
    except Exception:
        pass
    
    return list(best_h)
