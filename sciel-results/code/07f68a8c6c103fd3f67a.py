"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard (though this method is very fast)
    start_time = time.time()
    
    # Probe the function to determine if it's of the form y' = A y + b
    # Assuming constant coefficients (autonomous)
    
    # 1. Evaluate at y=0 to get b
    y_zero = np.zeros(n)
    f_zero = np.array(f(t0, y_zero), dtype=float)
    
    # 2. Evaluate at basis vectors to get columns of A
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_ej = np.array(f(t0, e_j), dtype=float)
        # f(t, y) = A y + b
        # A[:, j] = f(t, e_j) - f(t, 0)
        A[:, j] = f_ej - f_zero
        
    # Construct augmented matrix for the affine system
    # z' = M z, where z = [y; 1]
    # y' = A y + b
    # 1' = 0
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = f_zero  # b
    
    # Initial state z0
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Compute matrix exponential
    # expm(M * dt)
    E = expm(M * dt)
    
    # Final state
    z1 = E @ z0
    
    return z1[:n]
