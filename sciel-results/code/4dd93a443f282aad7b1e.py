"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    The peak of the autoconvolution of a non-negative function is at t=0, 
    given by R(0) = integral f(x)^2 dx. 
    Minimizing this peak subject to integral f = 1 and f >= 0 is equivalent 
    to minimizing the L2 norm of f. The optimal solution is a uniform 
    distribution over the support.
    
    We formulate this as a Quadratic Programming problem:
    Minimize sum(h[i]^2)
    Subject to sum(h[i]) = 1, h[i] >= 0.
    """
    # Initial guess: uniform
    x0 = np.ones(N) / N
    
    # Objective function: sum of squares
    def objective(h):
        return np.sum(h**2)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    
    # Bounds: h[i] >= 0
    bounds = [(0.0, None) for _ in range(N)]
    
    # Solve using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-12, 'maxiter': 1000})
    
    # Extract solution
    h_opt = res.x
    
    # Ensure non-negative and normalized (clean up numerical errors)
    h_opt = np.maximum(h_opt, 0)
    h_opt = h_opt / np.sum(h_opt)
    
    return list(h_opt)
# EVOLVE-BLOCK-END
