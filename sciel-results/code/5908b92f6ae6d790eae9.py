import time
import sys
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_ratio(heights):
    # heights is a list of N non-negative values
    h = np.array(heights)
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # Calculate L1 norm
    norm1 = np.sum(h)
    if norm1 == 0:
        return 1e10
        
    # Calculate autoconvolution
    # Using FFT for speed
    # Pad to 2N-1
    fft_size = 2 * N
    F = np.fft.fft(h, fft_size)
    G = F * F
    conv = np.fft.ifft(G).real
    
    # The convolution is defined on [-1/4, 1/4] + [-1/4, 1/4] = [-1/2, 1/2]
    # The center (t=0) is at index N-1?
    # Let's check indices.
    # h[0] is at x = -1/4 + 0.5 * dx
    # h[N-1] is at x = -1/4 + (N-0.5) * dx = 1/4 - 0.5 * dx
    # Convolution index k corresponds to sum i+j = k.
    # Center of mass is at N-1.
    # Max of conv
    max_conv = np.max(conv)
    
    ratio = max_conv / (norm1 * norm1)
    return ratio

def construct():
    # Try to optimize
    # Initial guess: Box
    x0 = np.ones(N)
    
    # Symmetry constraint: x[i] == x[N-1-i]
    # We can optimize only the first half and mirror it.
    half_N = N // 2
    x0_half = np.ones(half_N)
    
    def objective(x_half):
        # Mirror
        x_full = np.concatenate([x_half, x_half[::-1]])
        # If N is odd, handle middle? N=600 is even.
        return compute_ratio(x_full)
        
    # Use L-BFGS-B
    # Bounds: non-negative
    bounds = [(0, None)] * half_N
    
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    try:
        res = minimize(objective, x0_half, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-8})
        
        if time.time() > deadline:
            raise TimeoutError("Optimization took too long")
            
        best_half = res.x
        best_full = np.concatenate([best_half, best_half[::-1]])
        best_score = objective(best_half)
        
        # Compare with box
        box_score = 2.0 # Known value
        
        if best_score < box_score:
            return best_full.tolist()
        else:
            return [1.0] * N
            
    except Exception as e:
        # Fallback to box
        return [1.0] * N

# EVOLVE-BLOCK-END
