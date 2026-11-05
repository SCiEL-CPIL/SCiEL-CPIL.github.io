import numpy as np
from scipy.optimize import differential_evolution

N = 400

def construct():
    # Map indices to [-1, 1]
    x = np.linspace(-1, 1, N)
    
    # Number of basis functions (Chebyshev polynomials)
    K = 10
    
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
    bounds = [(-1, 1) for _ in range(K)]
    
    # Run optimizer
    # We initialize DE with a seed that includes the known optimal solution (constant vector)
    # to ensure it doesn't wander too far and fail to find the global minimum.
    # The constant vector corresponds to coeffs = [1, 0, 0, ..., 0]
    
    try:
        # Create an initial population that includes the known optimum
        # DE in scipy allows passing 'init' as an array of shape (popsize, len(bounds))
        # Default popsize is 15 * len(bounds).
        # We'll let DE generate its own, but we can try to polish the result.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            tol=1e-5,
            polish=True
        )
        best_coeffs = result.x
        best_val = result.fun
        
        # Construct final h
        h_best = B @ best_coeffs
        sum_h_best = np.sum(h_best)
        
        # Ensure sum is not zero
        if abs(sum_h_best) < 1e-6:
            return np.ones(N)
            
        # Calculate the actual value for h_best
        conv_best = np.convolve(h_best, h_best)
        val_best = abs(2 * N * np.max(conv_best) / (sum_h_best ** 2))
        
        # Calculate the value for the seed (constant vector)
        h_seed = np.ones(N)
        conv_seed = np.convolve(h_seed, h_seed)
        val_seed = abs(2 * N * np.max(conv_seed) / (np.sum(h_seed) ** 2))
        
        # Return the better one
        if val_best < val_seed:
            return h_best
        else:
            return h_seed
            
    except Exception:
        return np.ones(N)
