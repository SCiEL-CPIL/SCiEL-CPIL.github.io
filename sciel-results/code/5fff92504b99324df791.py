import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, use 120 to be safe
    
    def max_overlap(h):
        # Compute max overlap efficiently
        # overlap(s) = sum_i h_i * (1 - h_{(i+s)%N}) = N/2 - C(s)
        # where C(s) = sum_i h_i * h_{(i+s)%N}
        # Use FFT for circular autocorrelation
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        # C[s] = sum_i h[i] * h[(i-s)%N] (circular)
        # We want C(s) = sum_i h[i] * h[(i+s)%N] = C[-s % N]
        # So min_s C(s) = min(C)
        min_C = np.min(C)
        return N/2 - min_C
    
    x0 = np.full(N, 0.5)
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    best_h = x0.copy()
    best_val = max_overlap(x0)
    
    # Try SLSQP
    try:
        result = minimize(max_overlap, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if result.success or result.fun < best_val:
            h = np.clip(result.x, 0, 1)
            # Fix sum
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            val = max_overlap(h)
            if val < best_val:
                best_h = h
                best_val = val
    except:
        pass
    
    if time.time() > deadline - 10:
        return best_h
    
    # Try additional refinement with different initial points
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation
        np.random.seed(attempt)
        x0 = np.random.uniform(0, 1, N)
        x0 = x0 / np.sum(x0) * N/2
        x0 = np.clip(x0, 0, 1)
        
        try:
            result = minimize(max_overlap, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            h = np.clip(result.x, 0, 1)
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            val = max_overlap(h)
            if val < best_val:
                best_h = h
                best_val = val
        except:
            pass
    
    return best_h
