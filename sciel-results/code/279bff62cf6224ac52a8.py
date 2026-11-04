import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize

def construct():
    # Initial guess: roots of the standard Laguerre polynomial of order 12
    roots, _ = roots_laguerre(12)
    x0 = roots
    
    # Target sum of roots for L_12(x) is 2*12 + 1 = 25
    target_sum = 25.0
    
    # Objective function: Minimize the sum of squares (variance) 
    # subject to the sum being fixed.
    def objective(x):
        return np.sum(x**2)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum}
    ]
    
    # Bounds: roots must be positive
    bounds = [(1e-10, None) for _ in range(12)]
    
    # Optimize
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Extract optimized roots
    opt_roots = res.x
    
    # Ensure they are sorted
    opt_roots = np.sort(opt_roots)
    
    return [float(r) for r in opt_roots]
