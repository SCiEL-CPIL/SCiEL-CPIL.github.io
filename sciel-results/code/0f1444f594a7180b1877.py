import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess
    h0 = np.full(N, 0.5)
    
    # Objective function
    def objective(h):
        # h is flattened
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr)
        
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2}
    ]
    bounds = [(0, 1)] * N
    
    # Run optimization
    # Nelder-Mead is derivative-free and robust
    result = minimize(objective, h0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    h_opt = result.x
    
    # Ensure sum is exactly 50 and within bounds
    # Simple projection
    h_opt = np.clip(h_opt, 0, 1)
    diff = N / 2 - np.sum(h_opt)
    h_opt += diff / N
    h_opt = np.clip(h_opt, 0, 1)
    
    # Re-normalize if clipping changed sum
    # Iterative proportional fitting or just simple scaling if close
    # For safety, if sum is off, adjust slightly
    if abs(np.sum(h_opt) - N/2) > 1e-9 * N:
         # Fallback to seed if optimization failed badly
         return [0.5] * N
         
    return list(h_opt)
