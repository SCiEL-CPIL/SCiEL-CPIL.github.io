import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Grid setup
    h = 0.5 / N
    x_centers = -0.25 + (np.arange(N) + 0.5) * h
    t_centers = 4.0 * x_centers # Map to [-1, 1]
    
    K = 16 # Number of Legendre coefficients
    P_vals = np.polynomial.legendre.legvander(t_centers, K)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        a = params
        g = P_vals @ a
        f = g * g
        
        # Check for trivial solution
        if np.max(f) < 1e-10:
            return 1e10
            
        # Compute autoconvolution peak
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        autoconv = np.fft.irfft(F * F, n=n_fft)
        
        # Metric: Peak / (Integral * Max)
        # Integral approx = h * sum(f)
        # Peak approx = h * max(autoconv)
        # Ratio = (h * max(autoconv)) / (h * sum(f) * max(f)) = max(autoconv) / (sum(f) * max(f))
        
        denom = np.sum(f) * np.max(f)
        if denom < 1e-10:
            return 1e10
            
        return np.max(autoconv) / denom

    bounds = [(-1.0, 1.0) for _ in range(K)]
    
    best_val = 1.0 # Constant function value
    best_params = np.zeros(K)
    best_params[0] = 1.0 # Constant function g=1 -> f=1
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=150, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            init='sobol' # Better initialization
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    except Exception:
        pass

    if time.time() > deadline:
        # Return constant if timeout
        return [1.0] * N

    # Reconstruct best f
    g = P_vals @ best_params
    f = g * g
    
    # Optional: Local polish
    # We can try a few steps of L-BFGS-B on the same parameters
    # But DE usually gets close enough.
    
    return list(f)
# EVOLVE-BLOCK-END
