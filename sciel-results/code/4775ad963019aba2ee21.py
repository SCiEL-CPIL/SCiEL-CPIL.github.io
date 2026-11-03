import numpy as np
from scipy.optimize import differential_evolution

def construct():
    N = 50
    DELTA = 0.5 / N
    
    # Parameterization: h_i = u_i^2 / sum(u_j^2)
    # u is in [-1, 1]^N
    # This ensures h_i >= 0 and sum(h) = 1
    
    def objective(u):
        # Compute h
        h_sq = u**2
        sum_sq = np.sum(h_sq)
        if sum_sq < 1e-15:
            return 1e15
        h = h_sq / sum_sq
        
        # Calculate L2 squared
        l2_sq = np.sum(h**2)
        
        # Calculate Convolution Max
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        if l2_sq < 1e-15 or max_conv < 1e-15:
            return 1e15
            
        return max_conv / l2_sq

    # Bounds for u
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    # Run Differential Evolution
    # maxiter and popsize control the search depth
    # seed is fixed for reproducibility, but we can vary it if needed
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15,
        tol=1e-6,
        polish=True
    )
    
    u_opt = result.x
    h_opt_sq = u_opt**2
    sum_h_sq = np.sum(h_opt_sq)
    h_opt = h_opt_sq / sum_h_sq
    
    # Calculate final score
    l2_sq = np.sum(h_opt**2)
    h_conv = np.convolve(h_opt, h_opt, mode='full')
    max_conv = np.max(h_conv)
    
    # The score defined in the task is:
    # (sum h^2) / ( (sum h) * Delta * max_conv )
    # Since sum h = 1:
    # score = l2_sq / (Delta * max_conv)
    
    score = l2_sq / (DELTA * max_conv)
    
    return list(h_opt)
