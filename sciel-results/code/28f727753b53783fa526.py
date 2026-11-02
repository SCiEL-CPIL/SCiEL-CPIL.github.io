"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def objective(x):
        # Compute autoconvolution via FFT
        n_fft = 2 * N
        f = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(f * np.conj(f), n=n_fft)
        max_conv = np.max(conv[:2*N-1])
        s = np.sum(x)
        if s < 1e-12:
            return 1e10
        return max_conv / (s * s)
    
    # Start with uniform
    x0 = np.ones(N)
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try a few random restarts with slight perturbations
    rng = np.random.RandomState(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb from best so far
            noise = rng.randn(N) * 0.1 * np.max(best_x)
            x_start = np.maximum(best_x + noise, 0.0)
        
        bounds = [(0.0, None)] * N
        
        def callback(xk):
            nonlocal best_x, best_val
            if time.time() > deadline:
                raise ValueError("timeout")
            val = objective(xk)
            if val < best_val:
                best_val = val
                best_x = xk.copy()
        
        try:
            result = minimize(objective, x_start, method='L-BFGS-B',
                            bounds=bounds, callback=callback,
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except (ValueError, Exception):
            pass
    
    best_x = np.maximum(best_x, 0.0)
    
    # Final check
    final_val = objective(best_x)
    if final_val > best_val + 1e-10:
        best_x = x0
    
    return list(best_x)
# EVOLVE-BLOCK-END
