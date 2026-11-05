"""
Rewritten program: ODE solver for 15-dim stiff linear system using matrix exponential.
Assumes f(t, y) = A @ y where A is a constant matrix.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float).flatten()
    n = len(y0)
    
    # If dimension is 0, return empty array
    if n == 0:
        return np.array([])
        
    # Construct the matrix A by evaluating f on the standard basis vectors.
    # We assume f(t, y) = A @ y, so f(0, e_j) = A @ e_j = A[:, j]
    # We use t=0 for evaluation. For a "fixed" linear system, A is constant.
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    # Calculate the matrix exponential exp(A * (t1 - t0))
    # The solution to y' = A y is y(t) = exp(A * t) y(0)
    dt = t1 - t0
    if dt == 0:
        return y0.copy()
        
    M = scipy.linalg.expm(A * dt)
    
    # Apply the matrix exponential to the initial state
    y_end = M @ y0
    
    return y_end
# EVOLVE-BLOCK-END
