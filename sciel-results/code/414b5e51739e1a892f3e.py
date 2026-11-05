import numpy as np
from scipy.optimize import differential_evolution

N = 400

def construct():
    # Map indices to [-1, 1]
    x = np.linspace(-1, 1, N)
    
    # Number of basis functions (Chebyshev polynomials)
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
            return 1e6
        
        # Normalize h to have sum 1 to simplify the ratio to 2*N*max(conv)
        # Since value = 2*N*max(conv(h))/sum(h)^2, if we scale h such that sum(h)=1,
        # value = 2*N*max(conv(h)).
        # This is valid because the expression is homogeneous of degree 0.
        h_norm = h / sum_h
        
        conv_h = np.convolve(h_norm, h_norm)
        max_conv = np.max(conv_h)
        
        # We want to minimize 2 * N * max_conv
        return 2 * N * max_conv

    # Differential Evolution bounds
    # Coefficients can be positive or negative
    bounds = [(-10, 10) for _ in range(K)]
    
    # Run optimizer
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=30, 
            seed=42,
            tol=1e-5,
            polish=True,
            mutation=(0.5, 1.0)
        )
        best_coeffs = result.x
        best_val = result.fun
        
        # Construct final h
        h_best = B @ best_coeffs
        sum_h_best = np.sum(h_best)
        
        # Ensure sum is not zero
        if abs(sum_h_best) < 1e-6:
            return np.ones(N)
            
        return h_best
        
    except Exception:
        return np.ones(N)
