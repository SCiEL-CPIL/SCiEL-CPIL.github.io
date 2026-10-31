import numpy as np
from scipy.optimize import differential_evolution

N = 100

def construct():
    # We optimize d such that h = 0.5 + d
    # sum(d) = 0
    # bounds for d: [-0.5, 0.5]
    
    def objective(params):
        # params are the first N-1 values of d? 
        # To ensure sum=0, we can define d_N = -sum(d_1...d_{N-1})
        # But bounds on d_N might be violated.
        # Alternative: use a transformation that guarantees sum 0 and bounds.
        # Or just use penalty.
        
        d = np.zeros(N)
        d[:N-1] = params
        d[N-1] = -np.sum(params)
        
        # Check bounds
        if np.any(d < -0.5) or np.any(d > 0.5):
            return 1e6 # Penalty
            
        # Calculate autocorrelation of d
        # f(k) = sum d_i d_{i+k}
        # We want to maximize min(f(k))
        # So minimize -min(f(k))
        
        min_corr = 1e9
        for k in range(N):
            # Circular shift
            d_shifted = np.roll(d, k)
            corr = np.dot(d, d_shifted)
            if corr < min_corr:
                min_corr = corr
                
        return -min_corr

    # Initial guess: random d with sum 0
    # differential_evolution handles bounds
    bounds = [(-0.5, 0.5) for _ in range(N-1)]
    
    # Run optimization
    result = differential_evolution(objective, bounds, maxiter=50, popsize=15, seed=42)
    
    d = np.zeros(N)
    d[:N-1] = result.x
    d[N-1] = -np.sum(result.x)
    
    h = 0.5 + d
    
    # Ensure exact sum 50 (floating point error)
    h = h / np.sum(h) * (N/2)
    
    # Clip to [0,1] just in case, then re-normalize?
    # If d was within bounds, h is within [0,1].
    
    return h
