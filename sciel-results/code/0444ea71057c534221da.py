import numpy as np
from scipy.optimize import differential_evolution

N = 400

def construct():
    # Map indices to [-1, 1] for basis definition
    x = np.linspace(-1, 1, N)
    
    # Number of basis functions (Chebyshev polynomials)
    # Increased to 20 to allow more complex shapes
    K = 20
    
    # Precompute Chebyshev basis matrix
    # T_j(x) = cos(j * arccos(x))
    theta = np.arccos(x)
    B = np.cos(np.outer(np.arange(K), theta))
    
    # Objective function for differential evolution
    def objective(coeffs):
        h = B @ coeffs
        sum_h = np.sum(h)
        # If sum is close to zero, the value is huge (or undefined)
        if abs(sum_h) < 1e-6:
            return 1e10
        
        # Normalize h to have sum 1
        h_norm = h / sum_h
        
        conv_h = np.convolve(h_norm, h_norm)
        max_conv = np.max(conv_h)
        
        # We want to minimize 2 * N * max_conv
        return 2.0 * N * max_conv

    # Differential Evolution bounds
    # Coefficients can be positive or negative
    bounds = [(-1.0, 1.0) for _ in range(K)]
    
    # Run optimizer
    # Increased popsize and maxiter for better global search coverage
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=50, 
            seed=42,
            tol=1e-5,
            polish=True,
            strategy='best1bin'
        )
        best_coeffs = result.x
        
        # Construct final h
        h_best = B @ best_coeffs
        sum_h_best = np.sum(h_best)
        
        # Ensure sum is not zero
        if abs(sum_h_best) < 1e-6:
            return np.ones(N)
            
        return h_best
        
    except Exception:
        return np.ones(N)
