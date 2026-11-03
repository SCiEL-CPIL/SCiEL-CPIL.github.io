import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The goal is to minimize the upper bound on C4, which is approximated by minimizing
    the Dyson gas energy for the Laguerre potential.
    """
    N = 12
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # 1. Initialization: Dyson gas equilibrium approximation
    # For Laguerre, the density is supported on [0, infinity).
    # A good initial guess is the zeros of the Laguerre polynomial L_N^{(0)} or similar.
    # We can approximate this using np.polynomial.laguerre.laggauss
    # But laggauss gives nodes for integration, which are close to the roots of L_N.
    # Let's use the roots of the Laguerre polynomial of degree N as a starting point.
    # Note: The roots of L_N are positive.
    
    # Using numpy's laggauss to get initial points
    # laggauss(N) returns nodes and weights for the integral of e^{-x}
    # The nodes are the roots of L_N.
    try:
        x0, w = np.polynomial.laguerre.laggauss(N)
        x0 = np.sort(x0)
    except Exception:
        # Fallback to linear spacing
        x0 = np.linspace(1, N*2, N)
        
    # 2. Define the objective function and its gradient
    # Energy E = sum (x_i - ln(x_i)) - sum_{i<j} ln|x_i - x_j|
    # This corresponds to the potential V(x) = x - ln(x) and repulsion -ln|x_i-x_j|
    
    def objective_and_grad(x):
        if np.any(x <= 0) or np.any(np.diff(x) <= 0):
            return 1e10, np.zeros(N)
        
        E = 0.0
        grad = np.zeros(N)
        
        # Potential part: sum (x_i - ln(x_i))
        for i in range(N):
            E += x[i] - np.log(x[i])
            grad[i] += 1.0 - 1.0 / x[i]
            
        # Interaction part: - sum_{i<j} ln|x_i - x_j|
        for i in range(N):
            for j in range(i+1, N):
                diff = x[i] - x[j]
                if diff == 0:
                    return 1e10, np.zeros(N)
                E -= np.log(abs(diff))
                # d/dx_i (-ln|x_i - x_j|) = -1/(x_i - x_j)
                # d/dx_j (-ln|x_i - x_j|) = -1/(x_j - x_i) = 1/(x_i - x_j)
                grad[i] -= 1.0 / diff
                grad[j] += 1.0 / diff
                
        return E, grad

    # 3. Optimization
    # Use L-BFGS-B with bounds
    # Bounds: x_i > 0, and x_i < x_{i+1}
    # We can enforce ordering by optimizing over unconstrained variables and mapping,
    # or use bounds and hope the initial guess is ordered.
    # L-BFGS-B supports bounds.
    bounds = [(1e-6, None) for _ in range(N)]
    
    # We need to ensure x_i < x_{i+1}. 
    # If we just use bounds, it might violate ordering.
    # A robust way is to parameterize x_i = x_{i-1} + exp(y_i) or similar.
    # But L-BFGS-B works well if the initial guess is strictly ordered and the basin is convex-like.
    # The Dyson gas energy is convex for sufficiently separated particles? Not strictly.
    # Let's stick to the simple bounds and check ordering in the callback or post-process.
    # Actually, if we start with ordered roots and minimize a smooth function, they usually stay ordered
    # unless they collide. The repulsion term goes to +infinity as they collide, preventing it.
    
    res = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    # Check time
    if time.time() > deadline - safety_margin:
        return [float(xi) for xi in x0]

    x_opt = res.x
    
    # Ensure ordering and positivity
    x_opt = np.sort(x_opt)
    # Add small epsilon to ensure strict positivity if needed, though bounds handle it
    x_opt = np.maximum(x_opt, 1e-6)
    
    # Verify constraints
    if np.any(np.diff(x_opt) <= 0):
        # Fallback to initial guess if optimization failed
        return [float(xi) for xi in x0]
        
    return [float(xi) for xi in x_opt]
