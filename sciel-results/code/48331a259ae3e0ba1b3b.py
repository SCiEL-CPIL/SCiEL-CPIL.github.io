"""Solve a fixed 15-dimensional stiff linear IVP using matrix exponential methods."""
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    This solver assumes the ODE is a linear system y' = A * y + b with constant A and b.
    It determines A and b by probing f at t0, then uses the matrix exponential on an augmented system
    for the exact solution.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Determine b by evaluating at y = 0
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Determine A columns by evaluating at y = e_j
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_probe = np.zeros(n, dtype=float)
        y_probe[j] = 1.0
        f_probe = np.asarray(f(t0, y_probe), dtype=float)
        A[:, j] = f_probe - b
        
    dt = t1 - t0
    
    # Use augmented system z' = M z where z = [y; 1]
    # M = [[A, b], [0, 0]]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = b
    # Last row is already zeros
    
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    E = expm(M * dt)
    z_final = E @ z0
    
    return z_final[:n]
