import numpy as np
from scipy.optimize import differential_evolution

N = 400

def construct():
    # Basis functions
    i = np.arange(N)
    # Use first few harmonics
    n_harm = 5
    basis = [np.ones(N)]
    for k in range(1, n_harm + 1):
        basis.append(np.cos(2 * np.pi * k * i / N))
        basis.append(np.sin(2 * np.pi * k * i / N))
    
    def objective(coeffs):
        h = np.zeros(N)
        for c, b in zip(coeffs, basis):
            h += c * b
        S = np.sum(h)
        if abs(S) < 1e-6:
            return 1e6
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        return abs(2 * N * max_val / (S ** 2))
    
    # Bounds for coefficients
    # a0 (DC) should be large to keep S large
    # AC terms can be negative
    bounds = [(-10, 10)] + [(-1, 1)] * (2 * n_harm)
    
    # Run optimization
    # Use a seed for reproducibility in case of multiple runs? No, just run.
    # Time limit guard
    import time
    start_time = time.time()
    
    try:
        res = differential_evolution(objective, bounds, maxiter=100, popsize=10, seed=42, tol=1e-4)
        best_coeffs = res.x
    except Exception:
        best_coeffs = [1.0] + [0.0] * (2 * n_harm)
        
    h = np.zeros(N)
    for c, b in zip(best_coeffs, basis):
        h += c * b
        
    # Check time
    if time.time() - start_time > 100:
         # Fallback to seed if too slow? 
         pass
         
    return h
