"""
ODE solver for the 10x10 Heat Equation using Matrix Exponential.
"""
import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrates y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Assumes the system is linear and time-invariant (y' = A y), which is the case
    for the standard heat equation on a fixed grid. Computes the matrix A by
    evaluating f on basis vectors, then applies the matrix exponential.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Construct the system matrix A by evaluating f on the identity basis.
    # f(t, y) = A @ y
    A = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    dt = t1 - t0
    # Compute the matrix exponential exp(A * dt)
    M = expm(A * dt)
    
    # Apply to initial condition
    y_end = M @ y0
    
    return y_end
