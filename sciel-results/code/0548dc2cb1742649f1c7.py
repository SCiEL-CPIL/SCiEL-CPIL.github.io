import numpy as np
from scipy.optimize import differential_evolution

N = 400

def construct():
    # Map indices to [-1, 1]
    x = np.linspace(-1, 1, N)
    
    # Number of basis functions (Chebyshev polynomials)
    # Increased K to allow for more complex shapes
    K = 12
    
    # Precompute Chebyshev basis matrix
    # T_j(x) = cos(j * arccos(x))
    # Clip x to [-1, 1] to avoid numerical issues with arccos
    x_safe = np.clip(x, -1.0, 1.0)
    theta = np.arccos(x_safe)
    B = np.cos(np.outer(np.arange(K), theta))
    
    # Objective function for differential evolution
    def objective(coeffs):
        h = B @ coeffs
        sum_h = np.sum(h)
        
        # If sum is close to zero, the value is huge (or undefined)
        # Penalize heavily to guide search away
        if abs(sum_h) < 1e-8:
            return 1e10
            
        # Normalize h to have sum 1
        # The target value is abs(2 * N * max(conv(h)) / sum(h)^2)
        # With sum(h)=1, this becomes abs(2 * N * max(conv(h)))
        h_norm = h / sum_h
        
        conv_h = np.convolve(h_norm, h_norm)
        max_conv = np.max(conv_h)
        
        # Minimize the absolute value of the peak
        return abs(2.0 * N * max_conv)

    # Differential Evolution bounds
    # Since the objective is scale-invariant, the absolute magnitude of coeffs
    # doesn't matter, only their ratios. Wider bounds allow DE to explore
    # a wider range of ratios effectively.
    bounds = [(-10, 10) for _ in range(K)]
    
    # Run optimizer
    # Tuned for robustness and global search capability within time limit
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=20, 
            seed=42,
            tol=1e-5,
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.9
        )
        best_coeffs = result.x
        
        # Construct final h
        h_best = B @ best_coeffs
        sum_h_best = np.sum(h_best)
        
        # Ensure sum is not zero
        if abs(sum_h_best) < 1e-8:
            return np.ones(N)
            
        # The evaluator calculates value directly from h.
        # We return h_best.
        return h_best
        
    except Exception:
        return np.ones(N)
